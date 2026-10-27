// SPDX-License-Identifier: MIT
//! Bounded, explicitly requested performance-vote experiment; no production policy.
//! Public OGKM 610.57.04 ctrl2080perf.h:54-102 and nvos.h NVOS54.
//! Independent public-header C assertions verified all five Android CPU ABIs.
use crate::{abi, diag::diag, rm};
use nv_rm_core::conv::u64_from_usize;
use std::{os::fd::AsRawFd, rc::Rc};

#[repr(C)]
struct BoostParameters {
    flags: u32,
    duration: u32,
}

/// Owns one subdevice vote, bounded by RM to ten seconds even on abrupt exit.
/// Dropping the guard clears it; dropping the device also releases its RM owner.
pub struct PerformanceVote {
    device: Rc<rm::Device>,
    armed: bool,
}

impl PerformanceVote {
    pub fn acquire(device: Rc<rm::Device>) -> rm::Result<Self> {
        if device.client.version != "610.57.04" {
            return Err(rm::Error::Contract(
                "PERF_BOOST diagnostic wire is verified only for 610.57.04".into(),
            ));
        }
        device
            .client
            .compat
            .require_same_layout("NVOS54_PARAMETERS")?;
        let vote = Self {
            device,
            armed: true,
        };
        // Arm before issuing the call so a returned error still attempts CLEAR.
        vote.request(2, 10)?;
        Ok(vote)
    }

    fn request(&self, flags: u32, duration: u32) -> rm::Result<()> {
        let mut params = BoostParameters { flags, duration };
        let mut call = abi::Control {
            client: self.device.client.root,
            object: self.device.subdevice,
            command: 0x2080200a,
            parameters: u64_from_usize((&raw mut params).expose_provenance()),
            size: 8,
            ..Default::default()
        };
        // Exact-version gate plus independent C oracle above; stack payload and
        // owner stay live throughout the synchronous control request.
        rm::exchange(
            self.device.client.control.as_raw_fd(),
            0x2a,
            &mut call,
            "PERF_BOOST diagnostic",
        )?;
        rm::status("PERF_BOOST diagnostic", call.status)
    }

    pub fn clear(&mut self) -> rm::Result<()> {
        if self.armed {
            self.request(0, 0)?;
            self.armed = false;
        }
        Ok(())
    }
}

impl Drop for PerformanceVote {
    fn drop(&mut self) {
        if let Err(e) = self.clear() {
            diag!("PERF_BOOST cleanup: {e}; bounded expiry and RM object teardown remain active");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn payload_matches_public_610_c_oracle() {
        assert_eq!(size_of::<BoostParameters>(), 8);
        assert_eq!(align_of::<BoostParameters>(), 4);
        assert_eq!(std::mem::offset_of!(BoostParameters, flags), 0);
        assert_eq!(std::mem::offset_of!(BoostParameters, duration), 4);
        assert_eq!(size_of::<abi::Control>(), 32);
        assert_eq!(abi::iowr(0x2a, 32), 0xc020462a);
    }
}
