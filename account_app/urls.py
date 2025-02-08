from django.urls import path
from .views import UserRegistrationView, dashboard
from .forms import LoginForm
from django.contrib.auth import views as auth_views

app_name = 'account_app'

urlpatterns = [
    path('register/', UserRegistrationView.as_view(), name='register'),
    path('dashboard/', dashboard, name='dashboard'),
    path('login/', auth_views.LoginView.as_view(
        template_name='account_app/login.html', 
        form_class=LoginForm,
        redirect_authenticated_user=True
    ), name='login'),
    path('logout/', auth_views.LogoutView.as_view(
        next_page='account_app:login'
    ), name='logout'),
]