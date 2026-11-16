variable "aws_region" {
  description = "Região AWS para todos os recursos"
  type        = string
  default     = "sa-east-1"
}

variable "project_name" {
  description = "Prefixo de nome para todos os recursos da Fase 4"
  type        = string
  default     = "credit-guard-fase04"
}

variable "instance_type" {
  description = "Tipo da instância EC2 (t3.medium = 4GB RAM para API + Evidently + MLflow + Prometheus + Grafana)"
  type        = string
  default     = "t3.medium"
}

variable "dockerhub_image" {
  description = "Imagem Docker Hub da API de observabilidade e drift"
  type        = string
  default     = "techchallengefase02/credit-guard-api:latest"
}

variable "allowed_countries" {
  description = "Países permitidos pelo geo-bloqueio do WAF/CloudFront"
  type        = list(string)
  default     = ["BR", "PT"]
}

variable "base_domain" {
  description = "Domínio base gerenciado para os subdomínios HTTPS"
  type        = string
  default     = "credit.cloud-ip.cc"
}
