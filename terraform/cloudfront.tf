locals {
  cf_origin_id   = "EC2Origin"
  ec2_origin_dns = aws_instance.credit_guard.public_dns
  cert_arn       = aws_acm_certificate_validation.wildcard.certificate_arn
  waf_arn        = aws_wafv2_web_acl.credit_guard.arn
  all_methods    = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
  cached_methods = ["GET", "HEAD"]
}

resource "aws_cloudfront_distribution" "api" {
  enabled     = true
  aliases     = ["api.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "CreditGuard API — Fase 4"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 8000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type", "Accept"]
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-api"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "mlflow" {
  enabled     = true
  aliases     = ["mlflow.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "MLflow Tracking UI — Fase 4"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 5000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "all" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 30
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-mlflow"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "grafana" {
  enabled     = true
  aliases     = ["grafana.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "Grafana — Fase 4"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 3000
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "all" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 60
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-grafana"
    Project = var.project_name
  }
}

resource "aws_cloudfront_distribution" "prometheus" {
  enabled     = true
  aliases     = ["prometheus.${var.base_domain}"]
  web_acl_id  = local.waf_arn
  price_class = "PriceClass_100"
  comment     = "Prometheus — Fase 4"

  origin {
    domain_name = local.ec2_origin_dns
    origin_id   = local.cf_origin_id

    custom_origin_config {
      http_port              = 9090
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = local.cf_origin_id
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = local.all_methods
    cached_methods         = local.cached_methods
    compress               = true

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 30
  }

  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = var.allowed_countries
    }
  }

  viewer_certificate {
    acm_certificate_arn      = local.cert_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = {
    Name    = "${var.project_name}-cf-prometheus"
    Project = var.project_name
  }
}
