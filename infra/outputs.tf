output "x86_64_public_ip" {
  description = "Public IP of the c6i.large (x86_64) benchmark instance."
  value       = aws_instance.bench_x86.public_ip
}

output "arm64_public_ip" {
  description = "Public IP of the c6g.large (ARM64) benchmark instance."
  value       = aws_instance.bench_arm64.public_ip
}
