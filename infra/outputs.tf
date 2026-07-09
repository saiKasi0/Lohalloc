output "x86_64_public_ip" {
  description = "Public IP of the x86_64 benchmark instance (null when enable_x86 = false)."
  value       = var.enable_x86 ? aws_instance.bench_x86[0].public_ip : null
}

output "arm64_public_ip" {
  description = "Public IP of the ARM64 benchmark instance."
  value       = aws_instance.bench_arm64.public_ip
}

output "arm64_instance_type" {
  description = "Instance type of the ARM64 benchmark instance (used by infra/cloud_bench.sh for the local results directory name)."
  value       = var.arm_instance_type
}
