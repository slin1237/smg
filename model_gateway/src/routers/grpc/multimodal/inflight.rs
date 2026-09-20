//! Cap on the preprocessed media bytes the gateway holds in flight for engines.

use std::{sync::Arc, time::Duration};

use axum::response::Response;
use http::StatusCode;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{observability::metrics::Metrics, routers::error};

/// Granularity of the budget, so the permit count stays within the semaphore's range.
const UNIT_BYTES: usize = 1024;
/// How long a request waits for bytes to free up before it is refused.
const WAIT: Duration = Duration::from_secs(2);

/// Why a reservation was refused.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum InflightRefusal {
    /// Larger than the whole budget; waiting would never help.
    TooLarge,
    /// The budget did not free up in time.
    Busy,
}

/// Bytes of preprocessed media the gateway may hold in flight at once.
pub(crate) struct MultimodalInflight {
    budget_bytes: usize,
    units: usize,
    semaphore: Arc<Semaphore>,
    wait: Duration,
}

impl MultimodalInflight {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        let units = budget_bytes
            .div_ceil(UNIT_BYTES)
            .clamp(1, Semaphore::MAX_PERMITS);
        Self {
            budget_bytes,
            units,
            semaphore: Arc::new(Semaphore::new(units)),
            wait: WAIT,
        }
    }

    #[cfg(test)]
    fn with_wait(mut self, wait: Duration) -> Self {
        self.wait = wait;
        self
    }

    pub(crate) fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    /// Hold `bytes` of the budget until the returned permit drops.
    pub(crate) async fn reserve(&self, bytes: usize) -> Result<InflightPermit, InflightRefusal> {
        let units = bytes.div_ceil(UNIT_BYTES);
        if units > self.units {
            return Err(InflightRefusal::TooLarge);
        }
        let Ok(units) = u32::try_from(units) else {
            return Err(InflightRefusal::TooLarge);
        };
        let acquire = Arc::clone(&self.semaphore).acquire_many_owned(units);
        match tokio::time::timeout(self.wait, acquire).await {
            Ok(Ok(permit)) => Ok(InflightPermit { _permit: permit }),
            Ok(Err(_)) | Err(_) => Err(InflightRefusal::Busy),
        }
    }
}

/// Releases its share of the budget when dropped.
#[derive(Debug)]
pub(crate) struct InflightPermit {
    _permit: OwnedSemaphorePermit,
}

/// Reserve room for a request's media bytes, or the response that refuses it.
pub(crate) async fn reserve_multimodal_inflight(
    inflight: Option<&MultimodalInflight>,
    bytes: usize,
) -> Result<Option<InflightPermit>, Response> {
    let Some(inflight) = inflight else {
        return Ok(None);
    };
    match inflight.reserve(bytes).await {
        Ok(permit) => Ok(Some(permit)),
        Err(InflightRefusal::TooLarge) => {
            Metrics::record_admission_rejected("multimodal_too_large");
            Err(error::create_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                "multimodal_payload_too_large",
                format!(
                    "the request carries {bytes} bytes of preprocessed media, more than the {} bytes this gateway holds in flight",
                    inflight.budget_bytes()
                ),
            ))
        }
        Err(InflightRefusal::Busy) => {
            Metrics::record_admission_rejected("multimodal_inflight");
            Err(error::too_many_requests(
                "multimodal_inflight_budget",
                "the gateway is already holding its budget of preprocessed media in flight; retry shortly",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick(budget: usize) -> MultimodalInflight {
        MultimodalInflight::new(budget).with_wait(Duration::from_millis(20))
    }

    #[tokio::test]
    async fn bytes_are_held_until_the_permit_drops() {
        let inflight = quick(4096);
        let first = inflight.reserve(3000).await.unwrap();
        assert_eq!(
            inflight.reserve(2000).await.unwrap_err(),
            InflightRefusal::Busy
        );
        drop(first);
        assert!(inflight.reserve(2000).await.is_ok());
    }

    #[tokio::test]
    async fn a_request_above_the_whole_budget_is_refused_at_once() {
        let inflight = quick(4096);
        let started = std::time::Instant::now();
        assert_eq!(
            inflight.reserve(5000).await.unwrap_err(),
            InflightRefusal::TooLarge
        );
        assert!(started.elapsed() < Duration::from_millis(15));
        assert!(inflight.reserve(0).await.is_ok());
    }

    #[tokio::test]
    async fn refusals_answer_413_and_429_and_no_budget_means_no_permit() {
        assert!(reserve_multimodal_inflight(None, usize::MAX)
            .await
            .unwrap()
            .is_none());

        let inflight = quick(4096);
        let too_large = reserve_multimodal_inflight(Some(&inflight), 5000)
            .await
            .unwrap_err();
        assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let _held = inflight.reserve(4096).await.unwrap();
        let busy = reserve_multimodal_inflight(Some(&inflight), 1024)
            .await
            .unwrap_err();
        assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
