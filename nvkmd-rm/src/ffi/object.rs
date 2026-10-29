// SPDX-License-Identifier: MIT
//! The backend's objects behind the Mesa base pointers NVK holds: Mesa calls
//! an entry with a pointer to a base struct (`nvkmd_pdev`, `nvkmd_dev`,
//! `nvkmd_mem`, `nvkmd_va`, `nvkmd_ctx`), and the entry recovers the object
//! around it after checking that the base's ops pointer is this backend's
//! table for that type. The one recovery every object type uses lives here.
//!
//! Nothing here touches Mesa or the RM: the unit tests below run it, natively
//! and under Miri, over heap-built objects of a test type.
use core::ptr::NonNull;

/// A backend object NVK holds through a pointer to its Mesa base struct.
///
/// # Safety
/// `Self` is `#[repr(C)]` with a `Base` as its first field, and every `Base`
/// whose ops pointer ([`Object::ops`]) is [`Object::ours`] is the first field
/// of a `Self` this backend allocated with `Box` (Rust's global allocator,
/// `Self`'s layout): no other code hands out bases with that ops table.
pub(super) unsafe trait Object: Sized {
    /// Mesa's base struct, the object's first field.
    type Base;
    /// The ops table type the base points to.
    type Ops;
    /// The ops pointer of a base.
    fn ops(base: &Self::Base) -> *const Self::Ops;
    /// This backend's ops table for the type.
    fn ours() -> *const Self::Ops;
}

/// The object behind `base`, or `None` for NULL or another backend's.
///
/// # Safety
/// `base` is NULL or points to a live `T::Base` that no call mutates while
/// the reference lives.
pub(super) unsafe fn recover<'a, T: Object>(base: *mut T::Base) -> Option<&'a T> {
    // SAFETY: NULL or a live base (caller).
    let object = unsafe { ours::<T>(base) }?;
    // SAFETY: a base with T's ops is the first field of a live `T` (trait
    // contract), not mutated during 'a (caller).
    Some(unsafe { object.as_ref() })
}

/// The object behind `base` for exclusive use, or `None` for NULL or another
/// backend's.
///
/// # Safety
/// `base` is NULL or points to a live `T::Base` that no other call uses while
/// the reference lives (Mesa serialises calls per object).
pub(super) unsafe fn recover_mut<'a, T: Object>(base: *mut T::Base) -> Option<&'a mut T> {
    // SAFETY: NULL or a live base (caller).
    let mut object = unsafe { ours::<T>(base) }?;
    // SAFETY: a base with T's ops is the first field of a live `T` (trait
    // contract), used by this call alone during 'a (caller).
    Some(unsafe { object.as_mut() })
}

/// Takes back the object behind `base`, or `None` for NULL or another
/// backend's.
///
/// # Safety
/// `base` is NULL or points to a live `T::Base` that is not used again.
pub(super) unsafe fn take<T: Object>(base: *mut T::Base) -> Option<Box<T>> {
    // SAFETY: NULL or a live base (caller).
    let object = unsafe { ours::<T>(base) }?;
    // SAFETY: a base with T's ops is the first field of a `T` the backend
    // leaked from a `Box` (trait contract); the caller gives it up.
    Some(unsafe { Box::from_raw(object.as_ptr()) })
}

/// `base` as a pointer to its object when its ops are `T`'s.
///
/// # Safety
/// `base` is NULL or points to a live `T::Base`.
unsafe fn ours<T: Object>(base: *mut T::Base) -> Option<NonNull<T>> {
    let base = NonNull::new(base)?;
    // SAFETY: a live base (caller); the reference ends with the ops read.
    let ops = T::ops(unsafe { base.as_ref() });
    core::ptr::eq(ops, T::ours()).then(|| base.cast::<T>())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Ops {
        _id: u32,
    }
    static OURS: Ops = Ops { _id: 1 };
    static OTHER: Ops = Ops { _id: 2 };

    #[repr(C)]
    struct Base {
        ops: *const Ops,
    }

    #[repr(C)]
    struct Thing {
        base: Base,
        value: u64,
    }

    // SAFETY: `#[repr(C)]` with `Base` first; the tests below make every base
    // with `OURS` inside a boxed `Thing`.
    unsafe impl Object for Thing {
        type Base = Base;
        type Ops = Ops;
        fn ops(base: &Base) -> *const Ops {
            base.ops
        }
        fn ours() -> *const Ops {
            &raw const OURS
        }
    }

    fn leak(ops: *const Ops, value: u64) -> *mut Base {
        Box::into_raw(Box::new(Thing {
            base: Base { ops },
            value,
        }))
        .cast()
    }

    #[test]
    fn null_is_no_object() {
        // SAFETY: NULL.
        assert!(unsafe { recover::<Thing>(core::ptr::null_mut()) }.is_none());
        // SAFETY: NULL.
        assert!(unsafe { recover_mut::<Thing>(core::ptr::null_mut()) }.is_none());
        // SAFETY: NULL.
        assert!(unsafe { take::<Thing>(core::ptr::null_mut()) }.is_none());
    }

    #[test]
    fn our_objects_are_recovered_and_taken_back() {
        let base = leak(&raw const OURS, 7);
        // SAFETY: a live base, no other use meanwhile.
        let thing = unsafe { recover_mut::<Thing>(base) }.expect("our ops");
        thing.value += 1;
        // SAFETY: a live base, not mutated while the reference lives.
        let read = unsafe { recover::<Thing>(base) }.map(|thing| thing.value);
        assert_eq!(read, Some(8));
        // SAFETY: a live base, not used again.
        let owned = unsafe { take::<Thing>(base) }.expect("our ops");
        assert_eq!(owned.value, 8);
    }

    #[test]
    fn another_backends_base_is_refused_and_left_alone() {
        let mut foreign = Base {
            ops: &raw const OTHER,
        };
        let base = &raw mut foreign;
        // SAFETY: a live base on the stack, no other use meanwhile.
        assert!(unsafe { recover::<Thing>(base) }.is_none());
        // SAFETY: as above.
        assert!(unsafe { recover_mut::<Thing>(base) }.is_none());
        // SAFETY: as above; refused, so nothing is freed.
        assert!(unsafe { take::<Thing>(base) }.is_none());
    }
}
