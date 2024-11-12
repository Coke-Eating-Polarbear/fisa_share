from django import forms
from .models import UserProfile

class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['CustomerID', 'Pw', 'Email', 'username', 'Birth', 'SerialNum', 'Phone']

    def clean(self):
        cleaned_data = super().clean()
        serial_num = cleaned_data.get("SerialNum")

        # SerialNum 값에 따라 성별을 자동 설정합니다
        if serial_num == '1' or serial_num == '3':
            cleaned_data['sex'] = 'M'
        elif serial_num == '2' or serial_num == '4':
            cleaned_data['sex'] = 'F'
        else:
            self.add_error('SerialNum', '유효하지 않은 주민등록번호 뒷자리입니다.')

        return cleaned_data
