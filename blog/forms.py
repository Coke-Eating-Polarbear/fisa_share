from django import forms
from .models import UserProfile


class LoginForm(forms.Form):
    email = forms.EmailField(widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': '이메일'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '비밀번호'}))


class UserProfileForm(forms.ModelForm):
    password_confirm = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")

    class Meta:
        model = UserProfile
        fields = ['name', 'user_id', 'password', 'birth', 'serial', 'email']
        widgets = {
            'password': forms.PasswordInput(),
            'serial': forms.PasswordInput(),
        }

    def clean(self):
        cleaned_data = super().clean()
        
        # Password confirmation check
        password = cleaned_data.get("password")
        password_confirm = cleaned_data.get("password_confirm")
        if password != password_confirm:
            self.add_error('password_confirm', "Passwords do not match.")

        # Validate resident registration number format
        birth = cleaned_data.get("birth")
        serial = cleaned_data.get("serial")
        
        if birth and len(birth) != 6:
            self.add_error('birth', "The birth date part of the SSN must be 6 digits.")
        if serial and (len(serial) != 7 or serial[0] not in '1234'):
            self.add_error('serial', "The serial part of the SSN must be 7 digits and start with 1, 2, 3, or 4.")
        
        # 성별 설정
        if serial:
            if serial[0] in ['1', '3']:
                cleaned_data['sex'] = 'M'
            elif serial[0] in ['2', '4']:
                cleaned_data['sex'] = 'F'
            else:
                self.add_error('serial', "Invalid serial number.")

        return cleaned_data
