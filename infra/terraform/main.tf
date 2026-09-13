# Managed container deployment on AWS: ALB -> ECS Fargate (API) -> RDS PostgreSQL (pgvector) + S3 artifacts.
# No Kubernetes: one stateless service and two managed data stores do not justify a cluster (ADR-0028).
#   terraform init && terraform plan -var-file=envs/staging.tfvars

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 5.60" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  backend "s3" {} # configure per environment: bucket/key/region/dynamodb_table
}

provider "aws" {
  region = var.region
}

variable "region" { type = string }
variable "environment" { type = string } # staging | production
variable "image" { type = string }       # ghcr.io/owner/safety-assistant:1.2.3
variable "vpc_id" { type = string }
variable "private_subnets" { type = list(string) }
variable "public_subnets" { type = list(string) }
variable "certificate_arn" { type = string }
variable "oidc_issuer" { type = string }
variable "oidc_audience" { type = string }
variable "alert_email" { type = string }
variable "db_instance_class" {
  type    = string
  default = "db.t4g.medium"
}
variable "api_cpu" {
  type    = number
  default = 1024
}
variable "api_memory" {
  type    = number
  default = 3072 # fastembed model + in-memory BM25 index of ~20k chunks
}
variable "worker_count" {
  type    = number
  default = 1 # ingestion workers; scale by count, claims use SKIP LOCKED
}
variable "worker_memory" {
  type    = number
  default = 3072 # parsing a 4,000-page manual peaks around 1.5 GB
}
variable "oidc_client_id" { type = string }
variable "oidc_redirect_uri" { type = string }
variable "frontend_url" { type = string }
variable "desired_count" {
  type    = number
  default = 2
}

locals { name = "safety-assistant-${var.environment}" }

# ---------------------------------------------------------------- artifacts (immutable, content-addressed)
resource "aws_s3_bucket" "artifacts" { bucket = "${local.name}-artifacts" }
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

# ---------------------------------------------------------------- database (managed, TLS, backups, PITR)
resource "random_password" "db" {
  length  = 32
  special = false
}
resource "aws_secretsmanager_secret" "db" { name = "${local.name}/database-url" }
resource "random_password" "session" {
  length  = 48
  special = false
}
resource "aws_secretsmanager_secret" "session" { name = "${local.name}/session-secret" }
resource "aws_secretsmanager_secret_version" "session" {
  secret_id     = aws_secretsmanager_secret.session.id
  secret_string = random_password.session.result
}
# Value is set out-of-band (never in state or variables): aws secretsmanager put-secret-value ...
resource "aws_secretsmanager_secret" "oidc_client" { name = "${local.name}/oidc-client-secret" }
resource "aws_secretsmanager_secret_version" "db" {
  secret_id     = aws_secretsmanager_secret.db.id
  secret_string = "postgresql+psycopg://safety:${random_password.db.result}@${aws_db_instance.pg.address}:5432/safety_assistant?sslmode=require"
}
resource "aws_db_subnet_group" "pg" {
  name       = local.name
  subnet_ids = var.private_subnets
}
resource "aws_security_group" "db" {
  name   = "${local.name}-db"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}
resource "aws_db_parameter_group" "pg" {
  name   = "${local.name}-pg16"
  family = "postgres16"
  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }
}
resource "aws_db_instance" "pg" {
  identifier                   = local.name
  engine                       = "postgres"
  engine_version               = "16" # pgvector is a built-in RDS extension: CREATE EXTENSION vector
  instance_class               = var.db_instance_class
  allocated_storage            = 50
  max_allocated_storage        = 500
  storage_encrypted            = true
  db_name                      = "safety_assistant"
  username                     = "safety"
  password                     = random_password.db.result
  db_subnet_group_name         = aws_db_subnet_group.pg.name
  vpc_security_group_ids       = [aws_security_group.db.id]
  parameter_group_name         = aws_db_parameter_group.pg.name
  backup_retention_period      = 14 # daily snapshots + point-in-time recovery window
  deletion_protection          = var.environment == "production"
  multi_az                     = var.environment == "production"
  performance_insights_enabled = true
  skip_final_snapshot          = false
  final_snapshot_identifier    = "${local.name}-final"
}

# ---------------------------------------------------------------- compute (ECS Fargate, non-root image, read-only fs)
resource "aws_ecs_cluster" "this" { name = local.name }
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}"
  retention_in_days = 30
}
resource "aws_security_group" "api" {
  name   = "${local.name}-api"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 8010
    to_port         = 8010
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy" "task" {
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:HeadObject", "s3:ListBucket"], Resource = [aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*"] },
      { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = [aws_secretsmanager_secret.db.arn, aws_secretsmanager_secret.session.arn, aws_secretsmanager_secret.oidc_client.arn] },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = ["${aws_cloudwatch_log_group.api.arn}:*", "${aws_cloudwatch_log_group.worker.arn}:*"] },
    ]
  })
}
resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.task.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name                   = "api"
    image                  = var.image
    essential              = true
    user                   = "10001:10001"
    readonlyRootFilesystem = true
    portMappings           = [{ containerPort = 8010 }]
    environment = [
      { name = "APP_ENV", value = "production" },
      { name = "AUTH_MODE", value = "oidc" },
      { name = "OIDC_ISSUER", value = var.oidc_issuer },
      { name = "OIDC_AUDIENCE", value = var.oidc_audience },
      { name = "OIDC_CLIENT_ID", value = var.oidc_client_id },
      { name = "OIDC_REDIRECT_URI", value = var.oidc_redirect_uri },
      { name = "FRONTEND_URL", value = var.frontend_url },
      { name = "DEV_LOGIN_ENABLED", value = "false" },
      { name = "RERANKER", value = "cross_encoder" },
      { name = "ARTIFACT_STORE_URI", value = "s3://${aws_s3_bucket.artifacts.bucket}" },
      { name = "EMBEDDING_PROVIDER", value = "fastembed" },
      { name = "LLM_PROVIDER", value = "none" }, # evidence-only until a cleared provider is configured
      { name = "WEB_CONCURRENCY", value = "2" },
    ]
    secrets = [
      { name = "DATABASE_URL", valueFrom = aws_secretsmanager_secret.db.arn },
      { name = "SESSION_SECRET", valueFrom = aws_secretsmanager_secret.session.arn },
      { name = "OIDC_CLIENT_SECRET", valueFrom = aws_secretsmanager_secret.oidc_client.arn },
    ]
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8010/health/ready', timeout=5).status==200 else 1)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
    logConfiguration = {
      logDriver = "awslogs"
      options   = { awslogs-group = aws_cloudwatch_log_group.api.name, awslogs-region = var.region, awslogs-stream-prefix = "api" }
    }
  }])
}
resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  deployment_circuit_breaker {
    enable   = true
    rollback = true # failed deploys roll back to the previous task definition automatically
  }
  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200
  network_configuration {
    subnets         = var.private_subnets
    security_groups = [aws_security_group.api.id]
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8010
  }
}

# ---------------------------------------------------------------- worker (ingestion queue)
resource "aws_cloudwatch_log_group" "worker" {
  name              = "/${local.name}/worker"
  retention_in_days = 30
}
resource "aws_ecs_task_definition" "worker" {
  family                   = "${local.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.api_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.task.arn
  task_role_arn            = aws_iam_role.task.arn
  container_definitions = jsonencode([{
    name                   = "worker"
    image                  = var.image
    command                = ["worker"]
    essential              = true
    user                   = "10001:10001"
    readonlyRootFilesystem = true
    environment = [
      { name = "APP_ENV", value = "production" },
      { name = "AUTH_MODE", value = "oidc" },
      { name = "OIDC_ISSUER", value = var.oidc_issuer },
      { name = "OIDC_AUDIENCE", value = var.oidc_audience },
      { name = "ARTIFACT_STORE_URI", value = "s3://${aws_s3_bucket.artifacts.bucket}" },
      { name = "EMBEDDING_PROVIDER", value = "fastembed" },
      { name = "LLM_PROVIDER", value = "none" },
      { name = "DEV_LOGIN_ENABLED", value = "false" },
      { name = "MALWARE_SCANNER", value = "none" }, # point at a clamd service or managed scanner before accepting external uploads
    ]
    secrets = [
      { name = "DATABASE_URL", valueFrom = aws_secretsmanager_secret.db.arn },
      { name = "SESSION_SECRET", valueFrom = aws_secretsmanager_secret.session.arn },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options   = { awslogs-group = aws_cloudwatch_log_group.worker.name, awslogs-region = var.region, awslogs-stream-prefix = "worker" }
    }
  }])
}
resource "aws_ecs_service" "worker" {
  name            = "${local.name}-worker"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = var.worker_count
  launch_type     = "FARGATE"
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets         = var.private_subnets
    security_groups = [aws_security_group.api.id] # same egress rules: database + object storage only
  }
}

# ---------------------------------------------------------------- ingress
resource "aws_security_group" "alb" {
  name   = "${local.name}-alb"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_lb" "this" {
  name               = local.name
  load_balancer_type = "application"
  subnets            = var.public_subnets
  security_groups    = [aws_security_group.alb.id]
}
resource "aws_lb_target_group" "api" {
  name        = "${local.name}-api"
  port        = 8010
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  health_check {
    path     = "/health/ready"
    matcher  = "200"
    interval = 30
  }
}
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.this.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ---------------------------------------------------------------- alerting (SLO guardrails)
resource "aws_sns_topic" "alerts" { name = "${local.name}-alerts" }
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name          = "${local.name}-5xx"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
resource "aws_cloudwatch_metric_alarm" "alb_p95" {
  alarm_name          = "${local.name}-p95-latency"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 3
  threshold           = 2
  comparison_operator = "GreaterThanThreshold"
  dimensions          = { LoadBalancer = aws_lb.this.arn_suffix }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

output "alb_dns_name" { value = aws_lb.this.dns_name }
output "artifact_bucket" { value = aws_s3_bucket.artifacts.bucket }
