fn main() {
    std::process::Command::new("bash")
        .arg("./download.sh")
        .status()
        .unwrap();
}
