variable "aws_region" {
  description = "AWS region to provision the benchmark instances in."
  type        = string
  default     = "us-east-1"
}

variable "ssh_public_key" {
  description = "Public key (OpenSSH format) injected into both instances via cloud-init user_data, so CI can SSH in to run the benchmark suite. Required — set via TF_VAR_ssh_public_key or -var."
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed to SSH into the benchmark instances. Restrict this to the CI runner's IP range in production use; defaults to open for convenience during manual dispatch."
  type        = string
  default     = "0.0.0.0/0"
}

variable "arm_instance_type" {
  description = "Instance type for the ARM64 benchmark instance. Defaults to the CI baseline (c6g.large); override for one-off runs on bigger Graviton hardware (e.g. c8g.4xlarge)."
  type        = string
  default     = "c6g.large"
}

variable "x86_instance_type" {
  description = "Instance type for the x86_64 benchmark instance (when enabled)."
  type        = string
  default     = "c6i.large"
}

variable "enable_x86" {
  description = "Provision the x86_64 instance. CI runs both architectures (true); ARM-only one-off runs (infra/cloud_bench.sh) set this false so a single instance is billed."
  type        = bool
  default     = true
}
