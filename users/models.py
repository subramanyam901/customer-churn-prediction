from django.db import models


# Create your models here.
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100) # Used for 'Username' label
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    email = models.CharField(unique=True, max_length=100)
    # Background defaults for system compatibility
    mobile = models.CharField(max_length=100, default='0000000000')
    locality = models.CharField(max_length=100, default='N/A')
    address = models.CharField(max_length=1000, default='N/A')
    city = models.CharField(max_length=100, default='N/A')
    state = models.CharField(max_length=100, default='N/A')
    status = models.CharField(max_length=100, default='waiting')

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'


class UserActivity(models.Model):
    username = models.CharField(max_length=150)
    phone = models.CharField(max_length=20)
    plan = models.CharField(max_length=150)
    monthly_charges = models.FloatField()
    prediction_result = models.CharField(max_length=50)
    risk_score = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'UserActivity'