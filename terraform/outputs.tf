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

output "prometheus_url" {
  description = "URL do Prometheus (HTTPS via CloudFront)"
  value       = "https://prometheus.${var.base_domain}"
}

output "cloudfront_api_domain" {
  description = "Domínio CloudFront da API — criar CNAME api.credit.cloud-ip.cc → este valor"
  value       = aws_cloudfront_distribution.api.domain_name
}

output "cloudfront_mlflow_domain" {
  description = "Domínio CloudFront do MLflow — criar CNAME mlflow.credit.cloud-ip.cc → este valor"
  value       = aws_cloudfront_distribution.mlflow.domain_name
}

output "cloudfront_grafana_domain" {
  description = "Domínio CloudFront do Grafana — criar CNAME grafana.credit.cloud-ip.cc → este valor"
  value       = aws_cloudfront_distribution.grafana.domain_name
}

output "cloudfront_prometheus_domain" {
  description = "Domínio CloudFront do Prometheus — criar CNAME prometheus.credit.cloud-ip.cc → este valor"
  value       = aws_cloudfront_distribution.prometheus.domain_name
}

output "acm_validation_cname" {
  description = "CNAME de validação do certificado ACM para adicionar no ClouDNS"
  value = {
    name  = tolist(aws_acm_certificate.wildcard.domain_validation_options)[0].resource_record_name
    value = tolist(aws_acm_certificate.wildcard.domain_validation_options)[0].resource_record_value
  }
}
