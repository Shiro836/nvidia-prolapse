// SPDX-License-Identifier: MIT
//! Source-built Mesa SPIR-V/NIR/NAK bridge, loaded once and called serially.
//! This module never recognizes a fixture or substitutes a precompiled shader.
use crate::rm::{Error, Result};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_void},
    sync::{Mutex, OnceLock},
};

#[derive(Clone, Debug)]
pub struct Shader {
    pub code: Vec<u8>,
    pub registers: u32,
    pub shared_size: u32,
    pub local_size: u32,
    pub barriers: u32,
    pub workgroup: [u32; 3],
}
impl Shader {
    /// Resource limits checked before a shader can reach channel submission.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.workgroup[0] > 1024 || self.workgroup[1] > 1024 || self.workgroup[2] > 64 {
            return Err(Error::Contract(
                "shader workgroup dimension exceeds supported limits".into(),
            ));
        }
        let threads = self
            .workgroup
            .iter()
            .map(|v| u64::from(*v))
            .product::<u64>();
        let register_words = (u64::from(self.registers) * 32).next_multiple_of(256)
            * threads.div_ceil(32).next_multiple_of(4);
        if self.code.is_empty()
            || !self.code.len().is_multiple_of(16)
            || self.code.len() > 1048576
            || self.registers == 0
            || self.registers > 255
            || self.local_size != 0
            || self.shared_size > 32768
            || self.barriers > 16
            || self.workgroup.contains(&0)
            || threads > 1024
            || register_words > 65536
            || self.workgroup[0] > 1024
            || self.workgroup[1] > 1024
            || self.workgroup[2] > 64
        {
            return Err(Error::Contract("shader exceeds native code/register/shared/workgroup limits; local stacks are not implemented".into()));
        }
        Ok(())
    }
}
#[repr(C)]
#[derive(Default)]
struct Compiled {
    code: *mut u8,
    code_size: usize,
    registers: u32,
    local_bytes: u32,
    shared_bytes: u32,
    barriers: u32,
    workgroup: [u32; 3],
    cbuf_bytes: u32,
}
type Compile = unsafe extern "C" fn(
    *const u32,
    usize,
    *const c_char,
    u32,
    *mut Compiled,
    *mut c_char,
    usize,
) -> c_int;
type Free = unsafe extern "C" fn(*mut Compiled);
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
    fn dlclose(handle: *mut c_void) -> c_int;
}
struct Functions {
    compile: Compile,
    free: Free,
}
static COMPILER: OnceLock<std::result::Result<Mutex<Functions>, String>> = OnceLock::new();

fn functions() -> Result<&'static Mutex<Functions>> {
    let loaded=COMPILER.get_or_init(|| {
        let path=std::env::var("NV_NATIVE_COMPILER_PATH").unwrap_or_else(|_|"libnv_native_nak.so".into());
        let path=CString::new(path).map_err(|_|"compiler path contains NUL".to_string())?;
        // SAFETY: NUL strings and platform dynamic loader ABI. Handle is kept for
        // process lifetime; function pointers cannot outlive the loaded DSO.
        unsafe {
            let handle=dlopen(path.as_ptr(),2); // RTLD_NOW | RTLD_LOCAL
            if handle.is_null() {
                let error=dlerror();
                let message=if error.is_null(){"unknown loader error".into()}else{CStr::from_ptr(error).to_string_lossy().into_owned()};
                return Err(format!("load source-built NAK compiler: {message}; build native/compiler and set NV_NATIVE_COMPILER_PATH"));
            }
            let compile=dlsym(handle,c"nv_nak_compile".as_ptr());
            let free=dlsym(handle,c"nv_nak_free".as_ptr());
            if compile.is_null() || free.is_null() { dlclose(handle); return Err("NAK compiler is missing bridge entrypoints".into()); }
            Ok(Mutex::new(Functions{compile:std::mem::transmute::<*mut c_void,Compile>(compile),free:std::mem::transmute::<*mut c_void,Free>(free)}))
        }
    });
    loaded.as_ref().map_err(|e| Error::Contract(e.clone()))
}

pub fn compile(words: &[u32], entry: &CStr) -> Result<Shader> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return Err(Error::Contract("invalid SPIR-V header".into()));
    }
    let functions = functions()?
        .lock()
        .map_err(|_| Error::Contract("NAK compiler lock poisoned".into()))?;
    let mut result = Compiled::default();
    let mut error = [0 as c_char; 2048];
    // SAFETY: borrowed word/name buffers and correctly-sized C output structs;
    // serialized by the lock above, matching nv_nak.h in this source tree.
    let status = unsafe {
        (functions.compile)(
            words.as_ptr(),
            words.len(),
            entry.as_ptr(),
            120,
            &mut result,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if status != 0 {
        error[2047] = 0;
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        return Err(Error::Contract(format!(
            "SPIR-V compilation failed: {message}"
        )));
    }
    let shader = if result.code.is_null()
        || result.code_size == 0
        || result.code_size > 1048576
        || result.cbuf_bytes > 1152
    {
        Err(Error::Contract(
            "NAK returned code/constant metadata outside implemented limits".into(),
        ))
    } else {
        // SAFETY: bridge owns code_size initialized bytes until nv_nak_free.
        let code = unsafe { std::slice::from_raw_parts(result.code, result.code_size) }.to_vec();
        Ok(Shader {
            code,
            registers: result.registers,
            shared_size: result.shared_bytes,
            local_size: result.local_bytes,
            barriers: result.barriers,
            workgroup: result.workgroup,
        })
    };
    unsafe { (functions.free)(&mut result) };
    let shader = shader?;
    shader.validate()?;
    Ok(shader)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_c_abi_matches_independent_c_oracle() {
        // Independent real-header C probes: compiler/check.c and
        // build/gpu-coverage/compiler-layout-probe.py. GNU32/64 and all five
        // NDK r29 Android targets agree per pointer width; no driver calls.
        #[cfg(target_pointer_width = "64")]
        let expected = [48, 8, 0, 8, 16, 20, 24, 28, 32, 44];
        #[cfg(target_pointer_width = "32")]
        let expected = [40, 4, 0, 4, 8, 12, 16, 20, 24, 36];
        assert_eq!(
            [
                std::mem::size_of::<Compiled>(),
                std::mem::align_of::<Compiled>(),
                std::mem::offset_of!(Compiled, code),
                std::mem::offset_of!(Compiled, code_size),
                std::mem::offset_of!(Compiled, registers),
                std::mem::offset_of!(Compiled, local_bytes),
                std::mem::offset_of!(Compiled, shared_bytes),
                std::mem::offset_of!(Compiled, barriers),
                std::mem::offset_of!(Compiled, workgroup),
                std::mem::offset_of!(Compiled, cbuf_bytes),
            ],
            expected
        );
    }
    #[test]
    fn excessive_register_demand_is_rejected_before_submission() {
        let mut shader = Shader {
            code: vec![0; 16],
            registers: 255,
            shared_size: 0,
            local_size: 0,
            barriers: 0,
            workgroup: [1024, 1, 1],
        };
        assert!(shader.validate().is_err());
        shader.registers = 32;
        assert!(shader.validate().is_ok());
        shader.local_size = 16;
        assert!(shader.validate().is_err());
        shader.workgroup = [u32::MAX; 3];
        assert!(shader.validate().is_err());
    }
}
