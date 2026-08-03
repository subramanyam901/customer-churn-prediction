from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from users.models import UserRegistrationModel, UserActivity

def AdminLoginCheck(request):
    # This view is deprecated by Unified Auth logic in Users/views.py
    return redirect('index')

def AdminHome(request):
    if request.session.get('role') != 'admin': return redirect('index')
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/AdminHome.html', {'data': data})

def ViewRegisteredUsers(request):
    if request.session.get('role') != 'admin': return redirect('index')
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})

def AdminDeleteUser(request):
    if request.session.get('role') != 'admin': return redirect('index')
    uid = request.GET.get('uid')
    get_object_or_404(UserRegistrationModel, id=uid).delete()
    messages.warning(request, "User account removed from the system.")
    return redirect('ViewRegisteredUsers')

def AdminUpdateUser(request):
    if request.session.get('role') != 'admin': return redirect('index')
    if request.method == "POST":
        uid = request.POST.get('uid')
        user = get_object_or_404(UserRegistrationModel, id=uid)
        user.name = request.POST.get('name')
        user.email = request.POST.get('email')
        if request.POST.get('password'):
            user.password = request.POST.get('password')
        user.save()
        messages.success(request, f"Credentials for {user.loginid} updated.")
    return redirect('ViewRegisteredUsers')

def AdminReport(request):
    if request.session.get('role') != 'admin': return redirect('index')
    activities = UserActivity.objects.all().order_by('-timestamp')
    return render(request, 'admins/AdminReport.html', {'activities': activities})

def AdminDownloadReport(request):
    if request.session.get('role') != 'admin': return redirect('index')
    from django.http import HttpResponse
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    import pandas as pd
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="User_Activity_Report.pdf"'
    
    doc = SimpleDocTemplate(response, pagesize=landscape(A4), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    elements = []
    
    styles = getSampleStyleSheet()
    title = Paragraph("<b>User Activity Report</b>", styles['Heading1'])
    elements.append(title)
    
    timestamp = Paragraph(f"<i>Generated Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</i>", styles['Normal'])
    elements.append(timestamp)
    elements.append(Spacer(1, 0.2*inch))
    
    data = [['Username', 'Phone', 'Plan', 'Charges', 'Risk Score', 'Result', 'Date']]
    activities = UserActivity.objects.all().order_by('-timestamp')
    for act in activities:
        date_str = act.timestamp.strftime("%Y-%m-%d %H:%M")
        data.append([
            act.username,
            act.phone,
            act.plan,
            f"Rs. {act.monthly_charges:.2f}",
            f"{act.risk_score:.1f}",
            act.prediction_result,
            date_str
        ])
    
    t = Table(data, colWidths=[1.5*inch, 1.2*inch, 2*inch, 1*inch, 1*inch, 2*inch, 1.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
    ]))
    
    elements.append(t)
    doc.build(elements)
    
    return response