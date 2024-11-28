# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    id = models.BigAutoField(primary_key=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


class Average(models.Model):
    stageclass = models.CharField(db_column='StageClass', primary_key=True, max_length=10)  # Field name made lowercase. The composite primary key (StageClass, Inlevel) found, that is not supported. The first column is selected.
    inlevel = models.IntegerField(db_column='Inlevel')  # Field name made lowercase.
    spend = models.BigIntegerField(blank=True, null=True)
    income = models.BigIntegerField(blank=True, null=True)
    asset = models.BigIntegerField(blank=True, null=True)
    finance = models.BigIntegerField(blank=True, null=True)
    eat = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    transfer = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    utility = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    phone = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    home = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    hobby = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    fashion = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    party = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    allowance = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    study = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)
    medical = models.DecimalField(max_digits=4, decimal_places=2, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'average'
        unique_together = (('stageclass', 'inlevel'),)


class Card(models.Model):
    cardid = models.CharField(db_column='CardID', primary_key=True, max_length=256)  # Field name made lowercase.
    cardname = models.CharField(db_column='CardName', max_length=256)  # Field name made lowercase.
    corp = models.CharField(max_length=256, blank=True, null=True)
    benefits = models.CharField(db_column='Benefits', max_length=256)  # Field name made lowercase.
    image = models.CharField(db_column='Image', max_length=256)  # Field name made lowercase.
    details = models.TextField(db_column='Details')  # Field name made lowercase.
    url = models.CharField(db_column='URL', max_length=256)  # Field name made lowercase.
    cardtype = models.CharField(db_column='CardType', max_length=1)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'card'


class DProduct(models.Model):
    dsid = models.CharField(db_column='DSID', primary_key=True, max_length=256)  # Field name made lowercase.
    name = models.CharField(db_column='Name', max_length=256, blank=True, null=True)  # Field name made lowercase.
    bank = models.CharField(db_column='Bank', max_length=256, blank=True, null=True)  # Field name made lowercase.
    baser = models.FloatField(db_column='BaseR', blank=True, null=True)  # Field name made lowercase.
    maxir = models.FloatField(db_column='MaxIR', blank=True, null=True)  # Field name made lowercase.
    dstype = models.CharField(db_column='dsType', max_length=256, blank=True, null=True)  # Field name made lowercase.
    period = models.CharField(db_column='Period', max_length=256, blank=True, null=True)  # Field name made lowercase.
    amount = models.CharField(db_column='Amount', max_length=256, blank=True, null=True)  # Field name made lowercase.
    method = models.CharField(db_column='Method', max_length=256, blank=True, null=True)  # Field name made lowercase.
    customer = models.CharField(max_length=256, blank=True, null=True)
    benefits = models.CharField(db_column='Benefits', max_length=256, blank=True, null=True)  # Field name made lowercase.
    interestpay = models.TextField(db_column='InterestPay', blank=True, null=True)  # Field name made lowercase.
    notice = models.TextField(db_column='Notice', blank=True, null=True)  # Field name made lowercase.
    protect = models.CharField(db_column='Protect', max_length=256, blank=True, null=True)  # Field name made lowercase.
    conddesc = models.TextField(db_column='CondDesc', blank=True, null=True)  # Field name made lowercase.
    condit = models.TextField(blank=True, null=True)
    ratetype = models.CharField(db_column='RateType', max_length=256, blank=True, null=True)  # Field name made lowercase.
    dsname = models.CharField(max_length=256, blank=True, null=True)
    deep = models.CharField(max_length=256, blank=True, null=True)
    big_clu = models.CharField(max_length=256, blank=True, null=True)
    joincond = models.TextField(blank=True, null=True)
    cluster = models.IntegerField(db_column='Cluster', blank=True, null=True)  # Field name made lowercase.
    token = models.TextField(blank=True, null=True)
    mindate = models.IntegerField(blank=True, null=True)
    maxdate = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'd_product'


class DepositProduct(models.Model):
    product_name = models.CharField(max_length=255, blank=True, null=True)
    bank_name = models.CharField(max_length=255, blank=True, null=True)
    base_interest_rate = models.CharField(max_length=50, blank=True, null=True)
    max_interest_rate = models.CharField(max_length=50, blank=True, null=True)
    product_type = models.CharField(max_length=255, blank=True, null=True)
    duration = models.CharField(max_length=50, blank=True, null=True)
    amount = models.CharField(max_length=255, blank=True, null=True)
    join_method = models.CharField(max_length=255, blank=True, null=True)
    target = models.TextField(blank=True, null=True)
    preferential_condition = models.TextField(blank=True, null=True)
    interest_payment = models.CharField(max_length=255, blank=True, null=True)
    precautions = models.TextField(blank=True, null=True)
    deposit_protection = models.TextField(blank=True, null=True)
    conditions = models.TextField(blank=True, null=True)
    preferential_interest_condition = models.TextField(blank=True, null=True)
    interest_type = models.CharField(max_length=50, blank=True, null=True)
    review_required = models.CharField(max_length=50, blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    category = models.CharField(max_length=50, blank=True, null=True)
    combined_condition = models.TextField(blank=True, null=True)
    cluster = models.IntegerField(blank=True, null=True)
    tokenized_texts = models.TextField(blank=True, null=True)
    min_months = models.IntegerField(blank=True, null=True)
    max_months = models.IntegerField(blank=True, null=True)
    dsid = models.IntegerField(db_column='DSID', primary_key=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'deposit_product'


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    id = models.BigAutoField(primary_key=True)
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


class DsCustomer(models.Model):
    customerid = models.OneToOneField('Usertable', models.DO_NOTHING, db_column='CustomerID', primary_key=True)  # Field name made lowercase.
    amount = models.IntegerField(db_column='Amount')  # Field name made lowercase.
    period = models.IntegerField(db_column='Period')  # Field name made lowercase.
    category = models.CharField(db_column='Category', max_length=256)  # Field name made lowercase.
    benefits = models.CharField(db_column='Benefits', max_length=256)  # Field name made lowercase.
    wantspend = models.CharField(max_length=256)
    wantsave = models.CharField(max_length=256)
    ptype = models.IntegerField(db_column='PType')  # Field name made lowercase.
    term = models.IntegerField()
    banktype = models.IntegerField(db_column='BankType')  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'ds_customer'


class DsProduct(models.Model):
    dsid = models.CharField(db_column='DSID', primary_key=True, max_length=256)  # Field name made lowercase.
    bank = models.CharField(db_column='Bank', max_length=256)  # Field name made lowercase.
    baser = models.TextField(db_column='BaseR', blank=True, null=True)  # Field name made lowercase.
    maxir = models.TextField(db_column='MaxIR', blank=True, null=True)  # Field name made lowercase.
    dstype = models.CharField(db_column='DSType', max_length=256)  # Field name made lowercase.
    method = models.CharField(db_column='Method', max_length=256, blank=True, null=True)  # Field name made lowercase.
    benefits = models.CharField(db_column='Benefits', max_length=256)  # Field name made lowercase.
    interestpay = models.CharField(db_column='InterestPay', max_length=256)  # Field name made lowercase.
    notice = models.TextField(db_column='Notice')  # Field name made lowercase.
    protect = models.CharField(db_column='Protect', max_length=1)  # Field name made lowercase.
    conddesc = models.TextField(db_column='CondDesc', blank=True, null=True)  # Field name made lowercase.
    ratetype = models.CharField(db_column='RateType', max_length=256, blank=True, null=True)  # Field name made lowercase.
    dsname = models.CharField(db_column='DSname', max_length=256)  # Field name made lowercase.
    who = models.TextField(blank=True, null=True)
    how = models.CharField(db_column='How', max_length=255)  # Field name made lowercase.
    irbenefits = models.CharField(db_column='IRBenefits', max_length=255)  # Field name made lowercase.
    detail = models.CharField(db_column='Detail', max_length=255)  # Field name made lowercase.
    bktype = models.CharField(db_column='BkType', max_length=255)  # Field name made lowercase.
    condit = models.CharField(db_column='Condit', max_length=255)  # Field name made lowercase.
    minperiod = models.CharField(db_column='MinPeriod', max_length=20, blank=True, null=True)  # Field name made lowercase.
    maxperiod = models.IntegerField(db_column='MaxPeriod', blank=True, null=True)  # Field name made lowercase.
    maxamount = models.IntegerField(db_column='MaxAmount', blank=True, null=True)  # Field name made lowercase.
    minamount = models.IntegerField(db_column='MinAmount', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'ds_product'


class Favorite(models.Model):
    customerid = models.OneToOneField('Usertable', models.DO_NOTHING, db_column='CustomerID', primary_key=True)  # Field name made lowercase. The composite primary key (CustomerID, DSID) found, that is not supported. The first column is selected.
    dsid = models.ForeignKey(DsProduct, models.DO_NOTHING, db_column='DSID')  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'favorite'
        unique_together = (('customerid', 'dsid'),)


class MydataAsset(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase.
    income = models.IntegerField(db_column='Income', blank=True, null=True)  # Field name made lowercase.
    total = models.BigIntegerField(db_column='Total', blank=True, null=True)  # Field name made lowercase.
    estate = models.BigIntegerField(db_column='Estate', blank=True, null=True)  # Field name made lowercase.
    financial = models.BigIntegerField(db_column='Financial', blank=True, null=True)  # Field name made lowercase.
    ect = models.BigIntegerField(db_column='Ect', blank=True, null=True)  # Field name made lowercase.
    total_income = models.BigIntegerField(blank=True, null=True)
    monthly_income = models.BigIntegerField(blank=True, null=True)
    debt = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'mydata_asset'


class MydataDs(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase. The composite primary key (CustomerID, AccountID) found, that is not supported. The first column is selected.
    accountid = models.CharField(db_column='AccountID', max_length=256)  # Field name made lowercase.
    bankname = models.CharField(db_column='BankName', max_length=256, blank=True, null=True)  # Field name made lowercase.
    pname = models.CharField(db_column='PName', max_length=256, blank=True, null=True)  # Field name made lowercase.
    balance = models.BigIntegerField(db_column='Balance', blank=True, null=True)  # Field name made lowercase.
    dsrate = models.DecimalField(db_column='DSrate', max_digits=4, decimal_places=2, blank=True, null=True)  # Field name made lowercase.
    enddate = models.DateField(db_column='EndDate', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'mydata_ds'
        unique_together = (('customerid', 'accountid'),)


class MydataPay(models.Model):
    customerid = models.CharField(db_column='CustomerID', max_length=256)  # Field name made lowercase.
    pdate = models.DateField(db_column='Pdate', blank=True, null=True)  # Field name made lowercase.
    bizcode = models.CharField(db_column='Bizcode', max_length=256, blank=True, null=True)  # Field name made lowercase.
    store = models.CharField(db_column='Store', max_length=256, blank=True, null=True)  # Field name made lowercase.
    price = models.IntegerField(db_column='Price', blank=True, null=True)  # Field name made lowercase.
    type = models.CharField(db_column='Type', max_length=256, blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'mydata_pay'


class News(models.Model):
    norder = models.IntegerField(db_column='NOrder')  # Field name made lowercase.
    ndate = models.DateField(db_column='NDate')  # Field name made lowercase.
    title = models.CharField(db_column='Title', max_length=256)  # Field name made lowercase.
    content = models.TextField(db_column='Content')  # Field name made lowercase.
    url = models.CharField(db_column='URL', max_length=256)  # Field name made lowercase.
    summary = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'news'


class Recommend(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase.
    dsid = models.CharField(db_column='DSID', max_length=256, blank=True, null=True)  # Field name made lowercase.
    name = models.CharField(db_column='Name', max_length=256, blank=True, null=True)  # Field name made lowercase.
    bank = models.CharField(db_column='Bank', max_length=256, blank=True, null=True)  # Field name made lowercase.
    baser = models.FloatField(db_column='BaseR', blank=True, null=True)  # Field name made lowercase.
    maxir = models.FloatField(db_column='MaxIR', blank=True, null=True)  # Field name made lowercase.
    dstype = models.CharField(db_column='dsType', max_length=256, blank=True, null=True)  # Field name made lowercase.
    period = models.CharField(db_column='Period', max_length=256, blank=True, null=True)  # Field name made lowercase.
    amount = models.IntegerField(db_column='Amount', blank=True, null=True)  # Field name made lowercase.
    method = models.CharField(db_column='Method', max_length=256, blank=True, null=True)  # Field name made lowercase.
    customer = models.CharField(max_length=256, blank=True, null=True)
    benefits = models.CharField(db_column='Benefits', max_length=256, blank=True, null=True)  # Field name made lowercase.
    interestpay = models.TextField(db_column='InterestPay', blank=True, null=True)  # Field name made lowercase.
    notice = models.TextField(db_column='Notice', blank=True, null=True)  # Field name made lowercase.
    protect = models.CharField(db_column='Protect', max_length=256, blank=True, null=True)  # Field name made lowercase.
    conddesc = models.TextField(db_column='CondDesc', blank=True, null=True)  # Field name made lowercase.
    condit = models.TextField(blank=True, null=True)
    ratetype = models.CharField(db_column='RateType', max_length=256, blank=True, null=True)  # Field name made lowercase.
    dsname = models.CharField(max_length=256, blank=True, null=True)
    deep = models.CharField(max_length=256, blank=True, null=True)
    big_clu = models.CharField(max_length=256, blank=True, null=True)
    mindate = models.IntegerField(blank=True, null=True)
    maxdate = models.IntegerField(blank=True, null=True)
    cluster = models.IntegerField(db_column='Cluster', blank=True, null=True)  # Field name made lowercase.
    custom_num = models.IntegerField(blank=True, null=True)
    amount_num = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'recommend'


class SProduct(models.Model):
    dsid = models.IntegerField(db_column='DSID', primary_key=True)  # Field name made lowercase.
    productname = models.TextField(db_column='ProductName')  # Field name made lowercase.
    bankname = models.TextField(db_column='BankName')  # Field name made lowercase.
    baserate = models.FloatField(db_column='BaseRate')  # Field name made lowercase.
    maxpreferentialrate = models.FloatField(db_column='MaxPreferentialRate')  # Field name made lowercase.
    producttype = models.TextField(db_column='ProductType')  # Field name made lowercase.
    period = models.TextField(db_column='Period')  # Field name made lowercase.
    amount = models.TextField(db_column='Amount')  # Field name made lowercase.
    joinmethod = models.TextField(db_column='JoinMethod', blank=True, null=True)  # Field name made lowercase.
    target = models.TextField(db_column='Target', blank=True, null=True)  # Field name made lowercase.
    accumulationmethod = models.TextField(db_column='AccumulationMethod', blank=True, null=True)  # Field name made lowercase.
    preferentialconditions = models.TextField(db_column='PreferentialConditions', blank=True, null=True)  # Field name made lowercase.
    interestpayment = models.TextField(db_column='InterestPayment', blank=True, null=True)  # Field name made lowercase.
    precautions = models.TextField(db_column='Precautions', blank=True, null=True)  # Field name made lowercase.
    depositprotection = models.TextField(db_column='DepositProtection', blank=True, null=True)  # Field name made lowercase.
    review = models.TextField(db_column='Review', blank=True, null=True)  # Field name made lowercase.
    preferentialrateconditions = models.TextField(db_column='PreferentialRateConditions', blank=True, null=True)  # Field name made lowercase.
    preferentialconditiondescription = models.TextField(db_column='PreferentialConditionDescription', blank=True, null=True)  # Field name made lowercase.
    ratetype = models.TextField(db_column='RateType')  # Field name made lowercase.
    detaileddescription = models.TextField(db_column='DetailedDescription', blank=True, null=True)  # Field name made lowercase.
    category = models.TextField(db_column='Category')  # Field name made lowercase.
    minperiod = models.FloatField(db_column='MinPeriod', blank=True, null=True)  # Field name made lowercase.
    maxperiod = models.FloatField(db_column='MaxPeriod', blank=True, null=True)  # Field name made lowercase.
    maxamount = models.FloatField(db_column='MaxAmount', blank=True, null=True)  # Field name made lowercase.
    minamount = models.FloatField(db_column='MinAmount', blank=True, null=True)  # Field name made lowercase.
    periodmin = models.FloatField(db_column='PeriodMin', blank=True, null=True)  # Field name made lowercase.
    periodmax = models.FloatField(db_column='PeriodMax', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 's_product'


class SpendAmount(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase. The composite primary key (CustomerID, SDate) found, that is not supported. The first column is selected.
    sdate = models.CharField(db_column='SDate', max_length=20)  # Field name made lowercase.
    eat_amount = models.IntegerField(db_column='eat_Amount', blank=True, null=True)  # Field name made lowercase.
    transfer_amount = models.IntegerField(db_column='transfer_Amount', blank=True, null=True)  # Field name made lowercase.
    utility_amount = models.IntegerField(db_column='utility_Amount', blank=True, null=True)  # Field name made lowercase.
    phone_amount = models.IntegerField(db_column='Phone_Amount', blank=True, null=True)  # Field name made lowercase.
    home_amount = models.IntegerField(db_column='home_Amount', blank=True, null=True)  # Field name made lowercase.
    hobby_amount = models.IntegerField(db_column='hobby_Amount', blank=True, null=True)  # Field name made lowercase.
    fashion_amount = models.IntegerField(db_column='fashion_Amount', blank=True, null=True)  # Field name made lowercase.
    party_amount = models.IntegerField(db_column='party_Amount', blank=True, null=True)  # Field name made lowercase.
    allowance_amount = models.IntegerField(db_column='allowance_Amount', blank=True, null=True)  # Field name made lowercase.
    study_amount = models.IntegerField(db_column='study_Amount', blank=True, null=True)  # Field name made lowercase.
    medical_amount = models.IntegerField(db_column='medical_Amount', blank=True, null=True)  # Field name made lowercase.
    totalamount = models.IntegerField(db_column='TotalAmount', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'spend_amount'
        unique_together = (('customerid', 'sdate'),)


class SpendFreq(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase. The composite primary key (CustomerID, SDate) found, that is not supported. The first column is selected.
    sdate = models.CharField(db_column='SDate', max_length=20)  # Field name made lowercase.
    eat_freq = models.IntegerField(db_column='eat_Freq', blank=True, null=True)  # Field name made lowercase.
    transfer_freq = models.IntegerField(db_column='transfer_Freq', blank=True, null=True)  # Field name made lowercase.
    utility_freq = models.IntegerField(db_column='utility_Freq', blank=True, null=True)  # Field name made lowercase.
    phone_freq = models.IntegerField(db_column='Phone_Freq', blank=True, null=True)  # Field name made lowercase.
    home_freq = models.IntegerField(db_column='home_Freq', blank=True, null=True)  # Field name made lowercase.
    hobby_freq = models.IntegerField(db_column='hobby_Freq', blank=True, null=True)  # Field name made lowercase.
    fashion_freq = models.IntegerField(db_column='fashion_Freq', blank=True, null=True)  # Field name made lowercase.
    party_freq = models.IntegerField(db_column='party_Freq', blank=True, null=True)  # Field name made lowercase.
    allowance_freq = models.IntegerField(db_column='allowance_Freq', blank=True, null=True)  # Field name made lowercase.
    study_freq = models.IntegerField(db_column='study_Freq', blank=True, null=True)  # Field name made lowercase.
    medical_freq = models.IntegerField(db_column='medical_Freq', blank=True, null=True)  # Field name made lowercase.
    totalfreq = models.IntegerField(db_column='TotalFreq', blank=True, null=True)  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'spend_freq'
        unique_together = (('customerid', 'sdate'),)


class SubscribedProducts(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase.
    cardid = models.CharField(db_column='CardID', max_length=256)  # Field name made lowercase.
    category = models.CharField(db_column='Category', max_length=256)  # Field name made lowercase.
    dsid = models.CharField(db_column='DSID', max_length=256)  # Field name made lowercase.
    matdate = models.DateField(db_column='Matdate')  # Field name made lowercase.

    class Meta:
        managed = False
        db_table = 'subscribed_products'


class Usertable(models.Model):
    customerid = models.CharField(db_column='CustomerID', primary_key=True, max_length=256)  # Field name made lowercase.
    username = models.CharField(max_length=256)
    pw = models.CharField(db_column='Pw', max_length=256)  # Field name made lowercase.
    birth = models.DateField(db_column='Birth')  # Field name made lowercase.
    sex = models.CharField(db_column='Sex', max_length=1)  # Field name made lowercase.
    phone = models.CharField(db_column='Phone', max_length=11)  # Field name made lowercase.
    email = models.CharField(db_column='Email', max_length=256)  # Field name made lowercase.
    serialnum = models.CharField(db_column='SerialNum', max_length=256, blank=True, null=True)  # Field name made lowercase.
    stageclass = models.CharField(db_column='StageClass', max_length=1, blank=True, null=True)  # Field name made lowercase.
    inlevel = models.SmallIntegerField(db_column='Inlevel', blank=True, null=True)  # Field name made lowercase.
    num_primary_children = models.IntegerField(blank=True, null=True)
    num_secondary_children = models.IntegerField(blank=True, null=True)
    num_adult_children = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'usertable'


class Wc(models.Model):
    date = models.DateField(blank=True, null=True)
    image = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'wc'
