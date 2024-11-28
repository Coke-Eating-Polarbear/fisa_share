from django.db.models import UniqueConstraint
from django.db import models # type: ignore
from django.contrib.auth.hashers import make_password # type: ignore
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey


# UserProfile 모델 (회원 정보)
class UserProfile(models.Model):
    CustomerID = models.CharField(max_length=100, unique=True, primary_key=True)  # 아이디
    Pw = models.CharField(max_length=100)  # 비밀번호
    Email = models.EmailField(unique=True)  # 이메일
    username = models.CharField(max_length=100)  # 이름
    Birth = models.CharField(max_length=6)  # 생년월일
    SerialNum = models.CharField(max_length=1)  # 주민번호 뒷자리
    Phone = models.CharField(max_length=11)  # 전화번호
    sex = models.CharField(max_length=1, blank=True)  # 성별 (M, F)
    Stageclass = models.CharField(max_length=1)
    Inlevel = models.SmallIntegerField()

    def save(self, *args, **kwargs):
        # 비밀번호가 이미 해시되지 않은 경우에만 해시화
        if not self.Pw.startswith('pbkdf2_'):
            self.Pw = make_password(self.Pw)
        super().save(*args, **kwargs)

    def __str__(self):
        return self.CustomerID

    class Meta:
        db_table = 'usertable'


# Recommend 모델 (추천 상품 정보)
class Recommend(models.Model):
    CustomerID = models.ForeignKey(UserProfile, on_delete=models.CASCADE, db_column='CustomerID')
    dproduct = models.ForeignKey('DProduct', on_delete=models.CASCADE, null=True, blank=True, db_column='DProductID')
    sproduct = models.ForeignKey('SProduct', on_delete=models.CASCADE, null=True, blank=True, db_column='SProductID')

    class Meta:
        db_table = 'recommend'
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'dproduct'], name='unique_recommend_dproduct'),
            models.UniqueConstraint(fields=['CustomerID', 'sproduct'], name='unique_recommend_sproduct'),
        ]

    def __str__(self):
        return f"{self.CustomerID.CustomerID} - {self.dproduct or self.sproduct}"


# DsProduct 모델 (상품 정보)


# Wc 모델 (워드 클라우드 이미지 저장)
class Wc(models.Model):
    date = models.DateField()  # 날짜
    image = models.BinaryField()  # BLOB 형태의 이미지

    class Meta:
        db_table = 'wc'


# News 모델 (뉴스 정보)
class News(models.Model):
    norder = models.IntegerField()  # 뉴스 순서
    ndate = models.DateField()  # 뉴스 날짜
    title = models.CharField(max_length=256)  # 뉴스 제목
    content = models.TextField()  # 뉴스 내용
    url = models.CharField(max_length=256)  # 뉴스 URL
    summary = models.TextField()  # 뉴스 요약

    class Meta:
        db_table = 'news'


# Favorite 모델 (찜한 상품 정보)
class Favorite(models.Model):
    CustomerID = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        db_column='CustomerID'
    )
    # DProduct와 SProduct를 선택적으로 참조
    dproduct = models.ForeignKey('DProduct', on_delete=models.CASCADE, null=True, blank=True)
    sproduct = models.ForeignKey('SProduct', on_delete=models.CASCADE, null=True, blank=True)

    class Meta:
        db_table = 'favorite'
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'dproduct'], name='unique_favorite_dproduct'),
            models.UniqueConstraint(fields=['CustomerID', 'sproduct'], name='unique_favorite_sproduct'),
        ]

    def __str__(self):
        return f"{self.CustomerID.CustomerID} - {self.dproduct or self.sproduct}"
    

class Average(models.Model):
    stageclass = models.CharField(max_length=10, primary_key=True)  # StageClass 컬럼
    inlevel = models.IntegerField()  # Inlevel 컬럼
    spend = models.IntegerField()  # 소비
    income = models.IntegerField()  # 수입
    asset = models.IntegerField()  # 자산
    finance = models.IntegerField()  # 금융
    eat = models.DecimalField(max_digits=4, decimal_places=2)  # 식사
    transfer = models.DecimalField(max_digits=4, decimal_places=2)  # 교통
    utility = models.DecimalField(max_digits=4, decimal_places=2)  # 공과금
    phone = models.DecimalField(max_digits=4, decimal_places=2)  # 통신
    home = models.DecimalField(max_digits=4, decimal_places=2)  # 주거
    hobby = models.DecimalField(max_digits=4, decimal_places=2)  # 취미
    fashion = models.DecimalField(max_digits=4, decimal_places=2)  # 패션
    party = models.DecimalField(max_digits=4, decimal_places=2)  # 파티
    allowance = models.DecimalField(max_digits=4, decimal_places=2)  # 용돈
    study = models.DecimalField(max_digits=4, decimal_places=2)  # 학업
    medical = models.DecimalField(max_digits=4, decimal_places=2)  # 의료

    class Meta:
        db_table = 'average'  # 테이블 이름
        managed = False  # Django가 테이블을 생성/수정하지 않도록 설정
        unique_together = (('stageclass', 'inlevel'),)
        constraints = [
            models.UniqueConstraint(fields=['stageclass', 'inlevel'], name='unique_stage_inlevel')
        ]  # 복합 Primary Key 대체로 UniqueConstraint 사용

    def __str__(self):
        return f"{self.stageclass} - {self.inlevel}"

class spend(models.Model):
    CustomerID = models.CharField(max_length=256, primary_key=True)
    SDate = models.DateField()
    Category = models.CharField(max_length=256)
    Frequency = models.IntegerField()
    Amount = models.BigIntegerField()

    class Meta:
        db_table = 'spend'

class card(models.Model):
    cardID = models.CharField(max_length=256, primary_key=True)
    cardName = models.CharField(max_length=256)
    benefits = models.CharField(max_length=256)
    image = models.CharField(max_length=256)
    details = models.TextField()
    url = models.CharField(max_length=256)
    cardType = models.CharField(max_length=1)

    class Meta:
        db_table = 'card'

class MyDataAsset(models.Model):
    CustomerID = models.CharField(max_length=256, primary_key=True)  # CustomerID
    income = models.IntegerField()  # Income
    total = models.BigIntegerField()  # Total
    estate = models.BigIntegerField()  # Estate
    financial = models.BigIntegerField()  # Financial
    ect = models.BigIntegerField()  # Ect
    monthly_income = models.FloatField()
    financial = models.FloatField()
    debt = models.FloatField()
    total_income = models.FloatField()

    def __str__(self):
        return self.CustomerID
    
    class Meta:
        db_table = 'mydata_asset'
    
class MyDataDS(models.Model):
    CustomerID = models.CharField(max_length=256)  # CustomerID
    AccountID = models.CharField(max_length=256)  # AccountID
    bank_name = models.CharField(max_length=256)  # BankName
    pname = models.CharField(max_length=256)  # PName
    balance = models.BigIntegerField()  # Balance
    ds_rate = models.DecimalField(max_digits=4, decimal_places=2)  # DSRate
    end_date = models.DateField()  # EndDate

    class Meta:
        db_table = 'mydata_ds'  # 테이블 이름
        managed = False 
        unique_together = ('CustomerID', 'AccountID')  # 복합 기본키 설정
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'AccountID'], name='unique_customer_account')
        ] 

    def __str__(self):
        return f"{self.CustomerID} - {self.AccountID}"
    
class MyDataPay(models.Model):
    CustomerID = models.CharField(max_length=256)  # CustomerID
    pdate = models.DateField()  # Pdate
    bizcode = models.CharField(max_length=256)  # Bizcode
    store = models.CharField(max_length=256)  # Store
    price = models.IntegerField()  # Price
    type = models.CharField(max_length=256)  # Type

    def __str__(self):
        return f"{self.CustomerID} - {self.store} ({self.pdate})"
    
    class Meta:
        db_table = 'mydata_pay'

class SpendAmount(models.Model):
    CustomerID = models.CharField(max_length=256, primary_key=True)  # CustomerID
    SDate = models.CharField(max_length=20)  # SDate
    eat_amount = models.IntegerField(null=True, blank=True)  # eat_Amount
    transfer_amount = models.IntegerField(null=True, blank=True)  # transfer_Amount
    utility_amount = models.IntegerField(null=True, blank=True)  # utility_Amount
    phone_amount = models.IntegerField(null=True, blank=True)  # phone_Amount
    home_amount = models.IntegerField(null=True, blank=True)  # home_Amount
    hobby_amount = models.IntegerField(null=True, blank=True)  # hobby_Amount
    fashion_amount = models.IntegerField(null=True, blank=True)  # fashion_Amount
    party_amount = models.IntegerField(null=True, blank=True)  # party_Amount
    allowance_amount = models.IntegerField(null=True, blank=True)  # allowance_Amount
    study_amount = models.IntegerField(null=True, blank=True)  # study_Amount
    medical_amount = models.IntegerField(null=True, blank=True)  # medical_Amount
    TotalAmount = models.IntegerField(null=True, blank=True)  # TotalAmount

    class Meta:
        db_table = 'spend_amount'
        managed = False  # Django가 테이블을 생성하거나 수정하지 않음
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'SDate'], name='unique_amount_customer_sdate')
        ]  # 복합 키 대체로 설정

    def __str__(self):
        return f"{self.CustomerID} - {self.SDate}"
    
class DProduct(models.Model):
    dsid = models.CharField(max_length=50, primary_key=True)  # Primary Key 설정
    name = models.CharField(max_length=255, null=True, blank=True)
    bank = models.CharField(max_length=255, null=True, blank=True)
    baser = models.FloatField(null=True, blank=True)
    maxir = models.FloatField(null=True, blank=True)
    dtype = models.CharField(max_length=255, null=True, blank=True)
    period = models.CharField(max_length=255, null=True, blank=True)
    amount = models.CharField(max_length=255, null=True, blank=True)
    method = models.TextField(null=True, blank=True)
    customer = models.CharField(max_length=255, null=True, blank=True)
    benefits = models.TextField(null=True, blank=True)
    interestpay = models.TextField(null=True, blank=True)
    notice = models.TextField(null=True, blank=True)
    protect = models.TextField(null=True, blank=True)
    conddesc = models.TextField(null=True, blank=True)
    condit = models.TextField(null=True, blank=True)
    ratetype = models.CharField(max_length=255, null=True, blank=True)
    dsname = models.CharField(max_length=255, null=True, blank=True)
    deep = models.CharField(max_length=255, null=True, blank=True)
    big_clu = models.CharField(max_length=255, null=True, blank=True)
    joincond = models.TextField(null=True, blank=True)
    cluster = models.IntegerField(null=True, blank=True)
    token = models.CharField(max_length=255, null=True, blank=True)
    mindate = models.CharField(max_length=255, null=True, blank=True)
    maxdate = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = 'd_product'  # 기존 데이터베이스 테이블 이름


class SProduct(models.Model):
    DSID = models.AutoField(primary_key=True)  # int PK
    ProductName = models.TextField()  # text
    BankName = models.TextField()  # text
    BaseRate = models.FloatField()  # float
    MaxPreferentialRate = models.FloatField()  # float
    ProductType = models.TextField()  # text
    Period = models.TextField()  # text
    Amount = models.TextField()  # text
    JoinMethod = models.TextField()  # text
    Target = models.TextField()  # text
    AccumulationMethod = models.TextField()  # text
    PreferentialConditions = models.TextField()  # text
    InterestPayment = models.TextField()  # text
    Precautions = models.TextField()  # text
    DepositProtection = models.TextField()  # text
    Review = models.TextField()  # text
    PreferentialRateConditions = models.TextField()  # text
    PreferentialConditionDescription = models.TextField()  # text
    RateType = models.TextField()  # text
    DetailedDescription = models.TextField()  # text
    Category = models.TextField()  # text
    MinPeriod = models.FloatField()  # float
    MaxPeriod = models.FloatField()  # float
    MaxAmount = models.FloatField()  # float
    MinAmount = models.FloatField()  # float
    PeriodMin = models.FloatField()  # float
    PeriodMax = models.FloatField()  # float

    class Meta:
        db_table = 's_product'  # 기존 테이블 이름 유지

class SpendFreq(models.Model):
    CustomerID = models.CharField(max_length=256, primary_key=True)  # CustomerID
    SDate = models.CharField(max_length=20)  # SDate
    eat_Freq = models.IntegerField(null=True, blank=True)  # eat_Freq
    transfer_Freq = models.IntegerField(null=True, blank=True)  # transfer_Freq
    utility_Freq = models.IntegerField(null=True, blank=True)  # utility_Freq
    phone_Freq = models.IntegerField(null=True, blank=True)  # phone_Freq
    home_Freq = models.IntegerField(null=True, blank=True)  # home_Freq
    hobby_Freq = models.IntegerField(null=True, blank=True)  # hobby_Freq
    fashion_Freq = models.IntegerField(null=True, blank=True)  # fashion_Freq
    party_Freq = models.IntegerField(null=True, blank=True)  # party_Freq
    allowance_Freq = models.IntegerField(null=True, blank=True)  # allowance_Freq
    study_Freq = models.IntegerField(null=True, blank=True)  # study_Freq
    medical_Freq = models.IntegerField(null=True, blank=True)  # medical_Freq
    TotalFreq = models.IntegerField(null=True, blank=True)  # TotalFreq

    class Meta:
        db_table = 'spend_freq'
        managed = False  # Django가 테이블을 생성하거나 수정하지 않음
        constraints = [
            models.UniqueConstraint(fields=['CustomerID', 'SDate'], name='unique_freq_customer_sdate')
        ]  # 복합 키 대체로 설정

    def __str__(self):
        return f"{self.CustomerID} - {self.SDate}"