use std::fmt::Write;

/// Append `x` as Python's `repr(float)` would: shortest round-trip decimal, but
/// integer-valued finite floats keep a trailing `.0`.
pub fn push_pyrepr(buf: &mut String, x: f64) {
    if x.is_nan() {
        buf.push_str("nan");
        return;
    }
    if x.is_infinite() {
        buf.push_str(if x < 0.0 { "-inf" } else { "inf" });
        return;
    }
    let start = buf.len();
    let _ = write!(buf, "{x}");
    if !buf[start..].contains(['.', 'e', 'E']) {
        buf.push_str(".0");
    }
}
