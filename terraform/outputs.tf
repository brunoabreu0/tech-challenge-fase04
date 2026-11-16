output "ec2_instance_id" {
  description = "ID da instância EC2"
  value       = aws_instance.credit_guard.id
}

output "ec2_public_ip" {
  description = "IP público da instância EC2"
  value       = aws_instance.credit_guard.public_ip
}

output "api_url" {
  description = "URL da API CreditGuard (HTTPS via CloudFront)"
  value       = "https://api.${var.base_domain}"
}

output "mlflow_url" {
  description = "URL do MLflow Tracking Server (HTTPS via CloudFront)"
  value       = "https://mlflow.${var.base_domain}"
}

output "grafana_url" {
  description = "URL do Grafana Dashboard (HTTPS via CloudFront)"
  value       = "https://grafana.${var.base_domain}"
}
