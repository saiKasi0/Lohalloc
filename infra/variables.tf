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
