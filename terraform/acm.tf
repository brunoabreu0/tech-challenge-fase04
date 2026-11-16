resource "aws_acm_certificate" "wildcard" {
  provider          = aws.us_east_1
  domain_name       = "*.${var.base_domain}"
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name    = "${var.project_name}-wildcard-cert"
    Project = var.project_name
  }
}

resource "aws_acm_certificate_validation" "wildcard" {
  provider        = aws.us_east_1
  certificate_arn = aws_acm_certificate.wildcard.arn
}
