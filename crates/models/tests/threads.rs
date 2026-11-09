//! Its own test binary because rayon's global pool can only be set up once per process.

#[test]
fn set_threads_sizes_the_pool_and_rejects_zero_and_a_second_call() {
    assert!(rli_models::set_threads(0).is_err());
    rli_models::set_threads(3).unwrap();
    assert_eq!(rayon::current_num_threads(), 3);
    assert!(rli_models::set_threads(2).is_err());
    assert_eq!(rayon::current_num_threads(), 3);
}
