# Phase 6 hybrid-cloud benchmarking infrastructure.
#
# Provisions exactly 3 resources: one security group (SSH ingress) and two
# EC2 instances — c6i.large (x86_64) and c6g.large (ARM64) — so Lohalloc's
# criterion suite and native (LD_PRELOAD) harness run on real, dedicated
# hardware for both architectures without noisy-neighbor VM throttling.
#
# The SSH public key is injected via cloud-init `user_data` rather than a
# separate `aws_key_pair` resource, keeping this at exactly 3 resources (the
# two `data "aws_ami"` lookups don't count as resources).
#
# Usage (see CLAUDE.md and .github/workflows/bench.yml):
#   terraform -chdir=infra init
#   terraform -chdir=infra plan   -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"
#   terraform -chdir=infra apply  -auto-approve -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"
#   terraform -chdir=infra destroy -auto-approve  # ALWAYS destroy, even on failure

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Canonical's official AMI owner ID — required in provider v6+ alongside
# most_recent to avoid an ambiguous/untrusted AMI match.
locals {
  ubuntu_owner = "099720109477"
  cloud_init   = <<-EOT
    #cloud-config
    ssh_authorized_keys:
      - ${var.ssh_public_key}
  EOT
}

data "aws_ami" "ubuntu_amd64" {
  most_recent = true
  owners      = [local.ubuntu_owner]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "owner-id"
    values = [local.ubuntu_owner]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_ami" "ubuntu_arm64" {
  most_recent = true
  owners      = [local.ubuntu_owner]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*"]
  }
  filter {
    name   = "owner-id"
    values = [local.ubuntu_owner]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Resource 1/3: security group — SSH ingress only, all egress.
resource "aws_security_group" "bench" {
  name        = "lohalloc-bench"
  description = "Phase 6 hybrid-cloud benchmark instances: SSH in, everything out"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "lohalloc-bench"
    Purpose = "phase6-benchmarking"
  }
}

# Resource 2/3: x86_64 instance (skipped when enable_x86 = false, e.g.
# ARM-only one-off runs via infra/cloud_bench.sh).
resource "aws_instance" "bench_x86" {
  count                  = var.enable_x86 ? 1 : 0
  ami                    = data.aws_ami.ubuntu_amd64.id
  instance_type          = var.x86_instance_type
  vpc_security_group_ids = [aws_security_group.bench.id]
  user_data              = local.cloud_init

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
  }

  tags = {
    Name    = "lohalloc-bench-x86_64"
    Purpose = "phase6-benchmarking"
    Arch    = "x86_64"
  }
}

# Resource 3/3: ARM64 instance.
resource "aws_instance" "bench_arm64" {
  ami                    = data.aws_ami.ubuntu_arm64.id
  instance_type          = var.arm_instance_type
  vpc_security_group_ids = [aws_security_group.bench.id]
  user_data              = local.cloud_init

  root_block_device {
    volume_size = 40
    volume_type = "gp3"
  }

  tags = {
    Name    = "lohalloc-bench-arm64"
    Purpose = "phase6-benchmarking"
    Arch    = "arm64"
  }
}
