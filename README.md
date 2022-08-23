# arbitrary-int

This crate implements arbitrary numbers for Rust. Once included, you can use types like u5 or u120.

## Why yet another arbitrary integer crate?
There are quite a few similar crates to this one (the most famous being https://crates.io/crates/ux). After trying out a few of them I just realized that they are all very heavy: They create a ton of classes and take seconds to compile.

This crate is designed to be very short, using constant generics and only using macros lightly. Additionally, most of its functions are const, so that they can be used in const contexts.

## How to use
Unlike primitive data types like u32, there is no intrinsic syntax (Rust does not allow that). An instance is created as follows:

```rust
let value9 = u9::new(30);
```

This will create a value with 9 bits. If the value passed into new() doesn't fit, a panic! will be raised. This means that a function that accepts a u9 as an argument can be certain that its contents are never larger than an u9.

Standard operators are all overloaded, so it is possible to perform calculations using this type.

Internally, u9 will hold its data in an u16. It is possible to get this value:

```rust
let value9 = u9::new(30).value();
```

## Underlying data type
This crate defines types u1, u2, .., u126, u127 (obviously skipping the normal u8, u16, u32, u64, u128). Each of those types holds its actual data in the next larger data type (e.g. a u14 internally has an u16, a u120 internally has an u128). However, uXX are just type aliases; it is also possible to use the actual underlying generic struct:

```rust
let a = UInt::<u8, 5>::new(0b10101));
let b = UInt::<u32, 5>::new(0b10101));
```

In this example, a will have 5 bytes and be represented by a u8. This is identical to u5. b however is represented by a u32, so it is a different type from u5.

## Extract
A common source for arbitrary integers is by extracting them from bitfields. For example, if data contained 32 bits and we want to extract bits 4..=9, we could perform the following:

```rust
let a = u6::new((data >> 4) & 0b111111);
```

This is a pretty common operation, but it's easy to get it wrong: The number of 1s and u6 have to match. Also, new() will internally perform a bounds-check, which is unnecessary. Therefore, UInt provides an extract method:

```rust
let a = UInt::<u32, 6>::extract(data, 4);
```

This will result in an UInt::<u32, 6>. If desired, this can be reduced to a u6 (represented by a u8) like this:

```rust
let a: u6 = UInt::<u32, 6>::extract(data, 4).into();
```
