from django import template
from django.urls import reverse

register = template.Library()

@register.simple_tag(takes_context=True)
def get_home_url(context):
    user = context['request'].user
    if user.is_authenticated:
        if hasattr(user, 'doctor'):
            return reverse('doctor_app:dashboard')
        elif hasattr(user, 'patient'):
            return reverse('patient_app:upload')
    return reverse('account_app:login')