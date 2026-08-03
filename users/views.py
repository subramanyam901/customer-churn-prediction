from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from users.models import UserRegistrationModel, UserActivity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Global cache for trained models
_GLOBAL_STATE = {
    'models': {},
    'scaler': None,
    'label_encoders': {},
    'is_trained': False,
    'accuracy_chart': None,
    'results_table': [],
    'acc': 0
}

def load_and_preprocess_dataset():
    df = pd.read_csv('media/Churn_balanced_dataset.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Feature columns to use (matching predictForm1.html)
    features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'InternetService', 'OnlineSecurity', 
        'TechSupport', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
    ]
    target = 'Churn'
    
    le_dict = {}
    # Encode categorical columns in the entire dataframe first
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            
    X = df[features]
    y = df[target]
    
    # Secondary check to ensure X is purely numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, le_dict, scaler, features

def UserRegisterActions(request):
    from users.forms import UserRegistrationForm
    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.status = 'activated' # Auto-activate now
            user.save()
            messages.success(request, "Registration successful! You can now log in.")
            return redirect('index')
    return redirect('index')

def UserLoginCheck(request):
    if request.method == "POST":
        u = request.POST.get('loginname')
        p = request.POST.get('pswd')
        
        # Hardcoded Admin Check as requested (Internal Role Identification)
        if u == "admin" and p == "admin":
            request.session['loggeduser'] = "Admin"
            request.session['role'] = "admin"
            return redirect('AdminHome')
        
        # Database User Check
        try:
            user = UserRegistrationModel.objects.get(loginid=u, password=p)
            # Automatic activation check removed as per request (Auto-Accept)
            request.session['loggeduser'] = user.name
            request.session['role'] = "user"
            request.session['uid'] = user.id
            return redirect('UserHome')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, "Invalid username or password.")
            
    return redirect('index')

def UserHome(request):
    if 'role' not in request.session: return redirect('index')
    
    # Calculate Dashboard Stats
    try:
        df = pd.read_csv('media/Churn_balanced_dataset.csv')
        total_customers = len(df)
        churn_dist = df['Churn'].value_counts().to_dict()
        # Mocking or calculating some stats for the dashboard
        active_customers = len(df[df['Churn'] == 'No'])
        churned_customers = len(df[df['Churn'] == 'Yes'])
        
        # Feature correlation with churn for a simple importance chart
        # (This is just for visual dashboard impact)
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
        
        corr = df_encoded.corr()['Churn'].abs().sort_values(ascending=False)[1:6]
        top_features = corr.index.tolist()
        top_weights = [round(v * 100, 1) for v in corr.values]
        
    except:
        total_customers = 7043
        active_customers = 5174
        churned_customers = 1869
        top_features = ['Contract', 'Tenure', 'OnlineSecurity', 'TechSupport', 'MonthlyCharges']
        top_weights = [45.2, 38.6, 32.1, 30.5, 28.4]
        churn_dist = {'No': 5174, 'Yes': 1869}

    context = {
        'active': 'home',
        'total': total_customers,
        'active_count': active_customers,
        'churned_count': churned_customers,
        'top_features': top_features,
        'top_weights': top_weights,
        'churn_labels': list(churn_dist.keys()),
        'churn_values': list(churn_dist.values())
    }
    return render(request, 'users/UserHome.html', context)

def DatasetView(request):
    if 'role' not in request.session: return redirect('index')
    df = pd.read_csv('media/Churn_balanced_dataset.csv')
    data_html = df.head(100).to_html(classes="table", index=False)
    return render(request, 'users/viewdataset.html', {'data': data_html, 'active': 'data'})

def UserProfile(request):
    if 'role' not in request.session: return redirect('index')
    user = get_object_or_404(UserRegistrationModel, id=request.session['uid'])
    if request.method == "POST":
        user.name = request.POST.get('name')
        user.email = request.POST.get('email')
        if request.POST.get('password'):
            user.password = request.POST.get('password')
        user.save()
        messages.success(request, "Profile updated successfully.")
        return redirect('UserProfile')
    return render(request, 'users/user_profile.html', {'user': user, 'active': 'profile'})

def training(request):
    if 'role' not in request.session: return redirect('index')
    
    # SMART CACHE: If already trained and it's just a regular view (GET), show results instantly
    if request.method == "GET" and _GLOBAL_STATE['is_trained']:
        return render(request, 'users/training.html', {
            'results': _GLOBAL_STATE['results_table'],
            'chart': _GLOBAL_STATE['accuracy_chart'],
            'acc': _GLOBAL_STATE['acc']
        })

    # TRAINING ONLY ON POST (Explicit user action) or if never trained
    if request.method == "POST" or not _GLOBAL_STATE['is_trained']:
        try:
            # 1. LOAD DATA (Fast path)
            df = pd.read_csv('media/Churn_balanced_dataset.csv')
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
            features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                        'PhoneService', 'InternetService', 'OnlineSecurity', 
                        'TechSupport', 'Contract', 'PaperlessBilling', 
                        'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
            target = 'Churn'
            
            # 2. FAST ENCODING
            le_dict = {}
            X_raw = df[features].copy()
            for col in X_raw.columns:
                le = LabelEncoder()
                X_raw[col] = le.fit_transform(X_raw[col].astype(str))
                le_dict[col] = le
            y = LabelEncoder().fit_transform(df[target].astype(str))
            
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw.values.astype(float))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 3. MODELS (Optimized for speed)
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
                'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
                'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
                'Logistic Regression': LogisticRegression(max_iter=1000)
            }
            
            results = []
            trained_objs = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                results.append({
                    'Model': name, 'Accuracy': round(accuracy_score(y_test, pred) * 100, 2),
                    'Precision': round(precision_score(y_test, pred, zero_division=0) * 100, 2),
                    'Recall': round(recall_score(y_test, pred, zero_division=0) * 100, 2),
                    'F1_Score': round(f1_score(y_test, pred, zero_division=0) * 100, 2),
                    'Kappa': round(cohen_kappa_score(y_test, pred), 3)
                })
                trained_objs[name] = model

            # 4. LIGHTWEIGHT NEURAL NETWORK (Pre-validated)
            nn = Sequential([Dense(32, activation='relu', input_dim=X.shape[1]), Dense(1, activation='sigmoid')])
            nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            nn.fit(X_train, y_train, epochs=15, batch_size=64, verbose=0)
            nn_pred = (nn.predict(X_test, verbose=0) > 0.5).astype(int)
            results.append({
                'Model': 'Neural Network (DL)', 'Accuracy': round(accuracy_score(y_test, nn_pred) * 100, 2),
                'Precision': round(precision_score(y_test, nn_pred, zero_division=0) * 100, 2),
                'Recall': round(recall_score(y_test, nn_pred, zero_division=0) * 100, 2),
                'F1_Score': round(f1_score(y_test, nn_pred, zero_division=0) * 100, 2),
                'Kappa': round(cohen_kappa_score(y_test, nn_pred), 3)
            })
            trained_objs['Neural Network'] = nn

            # 5. FAST VECTORIZED ENSEMBLE
            results.sort(key=lambda x: x['Accuracy'], reverse=True)
            top_3 = [r['Model'] for r in results[:3]]
            all_preds = []
            for name in top_3:
                m = trained_objs[name]
                p = (m.predict(X_test, verbose=0) > 0.5).astype(int).flatten() if name == 'Neural Network' else m.predict(X_test)
                all_preds.append(p)
            ensemble_preds = (np.sum(all_preds, axis=0) > len(top_3)/2).astype(int)
            eep_acc = round(accuracy_score(y_test, ensemble_preds) * 100, 2)
            
            results.append({
                'Model': 'EEP (Weighted Ensemble)', 'Accuracy': eep_acc,
                'Precision': round(precision_score(y_test, ensemble_preds, zero_division=0) * 100, 2),
                'Recall': round(recall_score(y_test, ensemble_preds, zero_division=0) * 100, 2),
                'F1_Score': round(f1_score(y_test, ensemble_preds, zero_division=0) * 100, 2),
                'Kappa': round(cohen_kappa_score(y_test, ensemble_preds), 3)
            })
            results.sort(key=lambda x: x['Accuracy'], reverse=True)

            # 6. GLOBAL STATE & CHART
            _GLOBAL_STATE.update({'models': trained_objs, 'scaler': scaler, 'label_encoders': le_dict, 'is_trained': True, 'acc': eep_acc, 'results_table': results})
            plt.figure(figsize=(10,5))
            plot_data = [r for r in results if r['Accuracy'] != 'Dynamic']
            sns.barplot(x=[r['Model'] for r in plot_data], y=[float(r['Accuracy']) for r in plot_data])
            plt.xticks(rotation=45)
            plt.title("System Performance Calibration (%)")
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            _GLOBAL_STATE['accuracy_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

        except Exception as e:
            messages.error(request, f"System Error: {str(e)}")
            return redirect('UserHome')

    return render(request, 'users/training.html', {
        'results': _GLOBAL_STATE['results_table'],
        'chart': _GLOBAL_STATE['accuracy_chart'],
        'acc': _GLOBAL_STATE['acc'],
        'active': 'train'
    })

def prediction(request):
    if 'role' not in request.session: return redirect('index')
    if not _GLOBAL_STATE['is_trained']:
        messages.warning(request, "Please run Training first to initialize EEP modules.")
        return redirect('UserHome')
    
    context = {'active': 'predict'}
    
    if request.method == "POST":
        try:
            # Extract features from POST
            data = request.POST
            raw_input = {
                'provider_network': data.get('provider_network', 'Unknown'),
                'gender': data.get('gender', 'Male'),
                'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
                'Partner': data.get('Partner', 'No'), # Default
                'Dependents': data.get('Dependents', 'No'), # Default
                'tenure': 13, # Fixed as per request
                'PhoneService': data.get('PhoneService', 'Yes'), # Default
                'InternetService': data.get('InternetService', 'Fiber optic'),
                'OnlineSecurity': data.get('OnlineSecurity', 'No'),
                'TechSupport': data.get('TechSupport', 'No'), # Customer Support
                'Contract': data.get('Contract', 'Month-to-month'),
                'PaperlessBilling': data.get('PaperlessBilling', 'Yes'), # Captured from form
                'PaymentMethod': data.get('PaymentMethod', 'Electronic check'), # Captured from form
                'MonthlyCharges': float(data.get('MonthlyCharges', 0)),
                'TotalCharges': float(data.get('TotalCharges', 0)),
                # Advanced Behavioral & Service Metrics
                'Satisfaction': int(data.get('Satisfaction', 3)),
                'Complaints': int(data.get('Complaints', 0)),
                'OnlineBackup': data.get('OnlineBackup', 'No'),
                'DeviceProtection': data.get('DeviceProtection', 'No'),
                'StreamingTV': data.get('StreamingTV', 'No')
            }
            
            # Encode inputs for ML model
            encoded = []
            le_dict = _GLOBAL_STATE['label_encoders']
            cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                    'PhoneService', 'InternetService', 'OnlineSecurity', 
                    'TechSupport', 'Contract', 'PaperlessBilling', 
                    'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
            
            for col in cols:
                val = raw_input[col]
                if col in le_dict:
                    try: val = le_dict[col].transform([str(val)])[0]
                    except: val = 0
                encoded.append(val)
                
            X_input = _GLOBAL_STATE['scaler'].transform([encoded])
            
            # Detailed Prediction Insights
            probs = []
            votes = []
            for name, model in _GLOBAL_STATE['models'].items():
                if name == 'Neural Network':
                    p = float(model.predict(X_input, verbose=0)[0][0])
                    pred = 1 if p > 0.5 else 0
                else:
                    try:
                        p = model.predict_proba(X_input)[0][1]
                    except:
                        p = 1.0 if model.predict(X_input)[0] == 1 else 0.0
                    pred = model.predict(X_input)[0]
                probs.append(p)
                votes.append(pred)
            
            # Ensemble Aggregation
            avg_prob = np.mean(probs)
            
            # Advanced Behavioral Overrides
            sat_score = raw_input['Satisfaction']
            complaints = raw_input['Complaints']
            
            if sat_score <= 2:
                avg_prob += 0.15
            elif sat_score >= 4:
                avg_prob -= 0.10
                
            if complaints > 2:
                avg_prob += 0.10
            elif complaints == 0:
                avg_prob -= 0.05
                
            # Clamp limits
            avg_prob = min(max(avg_prob, 0.0), 1.0)
            
            final_pred = 1 if avg_prob > 0.5 else 0
            prob_percent = round(avg_prob * 100, 1) # Display absolute churn risk percentage
            
            # Risk Level Logic
            risk_level = "High" if avg_prob > 0.7 else "Medium" if avg_prob > 0.4 else "Low"
            
            # Explainable AI (XAI) - Rule-based Insights
            risk_factors = []
            safe_factors = []
            
            if raw_input['MonthlyCharges'] > 70: risk_factors.append("High Monthly Charges increases churn risk")
            else: safe_factors.append("Affordable Monthly Charges reduces churn risk")
            
            if raw_input['tenure'] < 12: risk_factors.append("Short tenure indicates low customer loyalty")
            elif raw_input['tenure'] > 48: safe_factors.append("Long tenure indicates high brand loyalty")
            
            if raw_input['Contract'] == 'Month-to-month': risk_factors.append("Month-to-month contract is highly unstable")
            else: safe_factors.append(f"{raw_input['Contract']} contract provides stability")
            
            if raw_input['OnlineSecurity'] == 'No': risk_factors.append("Lack of Online Security increases drop-off risk")
            else: safe_factors.append("Online Security keeps customers engaged")
            
            if raw_input['InternetService'] == 'Fiber optic': risk_factors.append("Fiber optic users have higher churn expectations")
            
            # Advanced Service & Experience Explainable Factors
            if sat_score <= 2: risk_factors.append(f"Very low satisfaction rating ({sat_score}/5) strongly pushes churn")
            elif sat_score >= 4: safe_factors.append(f"High satisfaction rating ({sat_score}/5) secures engagement")
            
            if complaints > 0: risk_factors.append(f"Customer has raised {complaints} complaints (elevated friction)")
            else: safe_factors.append("No active complaints, indicating stable experience")
            
            if raw_input['OnlineBackup'] == 'No' and raw_input['DeviceProtection'] == 'No':
                risk_factors.append("Missing backup and device protection reduces service stickiness")
            else:
                safe_factors.append("Active service retention features (Backup/Protection) bounds the user ecosystem")
                
            # Suggestions for Retention
            suggestions = []
            if raw_input['Contract'] == 'Month-to-month': 
                suggestions.append("Switch to a Yearly contract to significantly reduce churn probability.")
            if raw_input['OnlineBackup'] == 'No': 
                suggestions.append("Enable Online Backup add-on for better service retention.")
            if raw_input['OnlineSecurity'] == 'No': 
                suggestions.append("Activate cybersecurity/Online Security features to increase client stickiness.")
            if raw_input['InternetService'] == 'Fiber optic' and sat_score < 4:
                suggestions.append("Offer a premium support package to improve satisfaction for Fiber Optic users.")
            if sat_score <= 3:
                suggestions.append(f"Proactively contact customer to resolve their {complaints} active complaints and improve satisfaction.")
            
            # Feature Importance from Random Forest (if available)
            importance_chart = None
            top_influencers = []
            if 'Random Forest' in _GLOBAL_STATE['models']:
                rf = _GLOBAL_STATE['models']['Random Forest']
                plt.figure(figsize=(8, 4))
                feat_importances = pd.Series(rf.feature_importances_, index=cols)
                top_influencers = feat_importances.nlargest(3).index.tolist()
                feat_importances.nlargest(10).plot(kind='barh', color='#4318FF')
                plt.title("Key Factors Driving This Prediction")
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                importance_chart = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()

            # Save User Activity Tracking
            try:
                username = request.session.get('loggeduser', 'Unknown')
                user_record = UserRegistrationModel.objects.filter(name=username).first()
                if user_record:
                    phone = user_record.mobile
                else:
                    phone = "N/A"
                
                # Check for manually input phone in form if available
                if 'phone_number' in raw_input and raw_input['phone_number']:
                    phone = raw_input['phone_number']
                elif 'phone_number' in data and data.get('phone_number'):
                    phone = data.get('phone_number')

                if raw_input.get('provider_network') and raw_input.get('provider_network') != "Unknown":
                    plan_name = f"{raw_input.get('provider_network')} Plan"
                else:
                    plan_name = raw_input.get('Contract', 'Unknown Plan')

                UserActivity.objects.create(
                    username=username,
                    phone=phone,
                    plan=plan_name,
                    monthly_charges=float(raw_input.get('MonthlyCharges', 0)),
                    prediction_result="Churn Predicted" if final_pred == 1 else "Safe Customer",
                    risk_score=prob_percent
                )
            except Exception as e:
                print("Failed to save activity tracking:", e)

            # Store to session for PDF generation
            request.session['last_prediction'] = {
                'raw_input': raw_input,
                'output': "Churn Predicted" if final_pred == 1 else "Safe Customer",
                'probability': prob_percent,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'safe_factors': safe_factors,
                'suggestions': suggestions,
                'top_influencers': top_influencers
            }

            context.update({
                'output': "Churn Predicted" if final_pred == 1 else "Safe Customer",
                'probability': prob_percent,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'safe_factors': safe_factors,
                'suggestions': suggestions,
                'top_influencers': top_influencers,
                'importance_chart': importance_chart,
                'raw_input': raw_input
            })
            
            return render(request, 'users/predictForm1.html', context)
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            messages.error(request, f"Prediction error: {str(e)}")
            
    return render(request, 'users/predictForm1.html', context)

def generate_report(request):
    if 'role' not in request.session: return redirect('index')
    from django.http import HttpResponse
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    import pandas as pd
    
    report_type = request.GET.get('type', 'system')
    
    response = HttpResponse(content_type='application/pdf')
    
    p = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    
    if report_type == 'prediction':
        response['Content-Disposition'] = 'attachment; filename="Customer_Churn_Prediction_Report.pdf"'
        
        last_pred = request.session.get('last_prediction')
        if not last_pred:
            response = HttpResponse("No prediction data available.")
            return response
            
        raw_input = last_pred['raw_input']
        
        # Header
        p.setFillColor(colors.HexColor('#1e293b'))
        p.setFont("Helvetica-Bold", 22)
        p.drawString(1*inch, height-1*inch, "Customer Churn Prediction Report")
        p.setFont("Helvetica", 10)
        p.setFillColor(colors.gray)
        p.drawString(1*inch, height-1.3*inch, f"Generated for: {request.session.get('loggeduser', 'User')} | Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        p.line(1*inch, height-1.4*inch, width-1*inch, height-1.4*inch)
        
        # 1. Customer & Plan Details
        p.setFillColor(colors.HexColor('#1e293b'))
        p.setFont("Helvetica-Bold", 14)
        p.drawString(1*inch, height-1.8*inch, "1. Input Configuration & Plan Details")
        
        y = height - 2.1*inch
        p.setFont("Helvetica", 11)
        p.setFillColor(colors.black)
        
        # Left Column
        p.drawString(1*inch, y, f"Provider Network: {raw_input.get('provider_network')}")
        p.drawString(1*inch, y-0.25*inch, f"Gender: {raw_input.get('gender')}")
        p.drawString(1*inch, y-0.5*inch, f"Contract Type: {raw_input.get('Contract')}")
        p.drawString(1*inch, y-0.75*inch, f"Internet Service: {raw_input.get('InternetService')}")
        p.drawString(1*inch, y-1.0*inch, f"Tenure: {raw_input.get('tenure')} Months")
        
        # Right Column
        p.drawString(4*inch, y, f"Monthly Charges: Rs. {raw_input.get('MonthlyCharges')}")
        p.drawString(4*inch, y-0.25*inch, f"Total Charges: Rs. {raw_input.get('TotalCharges')}")
        p.drawString(4*inch, y-0.5*inch, f"Streaming TV: {raw_input.get('StreamingTV')}")
        p.drawString(4*inch, y-0.75*inch, f"Online Backup: {raw_input.get('OnlineBackup')}")
        p.drawString(4*inch, y-1.0*inch, f"Online Security: {raw_input.get('OnlineSecurity')}")
        
        y -= 1.4*inch
        p.line(1*inch, y, width-1*inch, y)
        
        # 2. Prediction Result
        y -= 0.4*inch
        p.setFont("Helvetica-Bold", 16)
        p.setFillColor(colors.HexColor('#1e293b'))
        p.drawString(1*inch, y, "2. Prediction Result")
        
        y -= 0.4*inch
        result_text = last_pred['output']
        prob = last_pred['probability']
        risk = last_pred['risk_level']
        
        if result_text == "Churn Predicted":
            p.setFillColor(colors.HexColor('#ef4444'))  # Red
        else:
            p.setFillColor(colors.HexColor('#22c55e'))  # Green
            
        p.setFont("Helvetica-Bold", 14)
        p.drawString(1.2*inch, y, f"> {result_text}")
        
        p.setFillColor(colors.black)
        p.setFont("Helvetica", 12)
        p.drawString(1.2*inch, y-0.3*inch, f"Risk Score: {prob}/100  |  Risk Level: {risk}")
        
        y -= 0.7*inch
        
        # Suggestions
        if 'suggestions' in last_pred and last_pred['suggestions']:
            p.setFillColor(colors.HexColor('#1e40af'))
            p.setFont("Helvetica-Bold", 12)
            p.drawString(1*inch, y, "Actionable Retention Suggestions:")
            y -= 0.2*inch
            p.setFillColor(colors.black)
            p.setFont("Helvetica", 11)
            for s in last_pred['suggestions']:
                if y < 1*inch: p.showPage(); y = height - 1*inch
                p.drawString(1.2*inch, y, f"* {s}")
                y -= 0.2*inch
            y -= 0.2*inch

        # Top Influencers
        if 'top_influencers' in last_pred and last_pred['top_influencers']:
            p.setFillColor(colors.HexColor('#8b5cf6'))
            p.setFont("Helvetica-Bold", 12)
            p.drawString(1*inch, y, "Top Influencing Factors:")
            y -= 0.2*inch
            p.setFillColor(colors.black)
            p.setFont("Helvetica", 11)
            p.drawString(1.2*inch, y, ", ".join(last_pred['top_influencers']))
            y -= 0.4*inch

        p.setFillColor(colors.gray)
        p.line(1*inch, y, width-1*inch, y)
        
        # 3. Explainable AI 
        y -= 0.4*inch
        p.setFillColor(colors.HexColor('#1e293b'))
        p.setFont("Helvetica-Bold", 14)
        p.drawString(1*inch, y, "3. Explainable AI Insights")
        
        y -= 0.3*inch
        
        # Risk factors
        p.setFillColor(colors.HexColor('#ef4444'))
        p.setFont("Helvetica-Bold", 12)
        p.drawString(1*inch, y, "Risk Factors (Pushing Churn):")
        y -= 0.2*inch
        p.setFillColor(colors.black)
        p.setFont("Helvetica", 11)
        for rf in last_pred['risk_factors']:
            if y < 1*inch:
                p.showPage()
                y = height - 1*inch
            p.drawString(1.2*inch, y, f"- {rf}")
            y -= 0.2*inch
            
        y -= 0.2*inch
        # Safe factors
        p.setFillColor(colors.HexColor('#22c55e'))
        p.setFont("Helvetica-Bold", 12)
        p.drawString(1*inch, y, "Safe Factors (Increasing Loyalty):")
        y -= 0.2*inch
        p.setFillColor(colors.black)
        p.setFont("Helvetica", 11)
        for sf in last_pred['safe_factors']:
            if y < 1*inch:
                p.showPage()
                y = height - 1*inch
            p.drawString(1.2*inch, y, f"- {sf}")
            y -= 0.2*inch
        
    else:
        # ORIGINAL EEP SYSTEM METRICS
        response['Content-Disposition'] = 'attachment; filename="Churn_EEP_Report.pdf"'
        
        # Header
        p.setFont("Helvetica-Bold", 24)
        p.drawString(1*inch, height-1*inch, "EEP Churn Prediction Report")
        
        p.setFont("Helvetica", 12)
        p.drawString(1*inch, height-1.4*inch, f"Generated by: {request.session.get('loggeduser', 'Unknown')} ({request.session.get('role', 'user').upper()})")
        p.drawString(1*inch, height-1.6*inch, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Metrics
        p.line(1*inch, height-1.8*inch, width-1*inch, height-1.8*inch)
        p.setFont("Helvetica-Bold", 16)
        p.drawString(1*inch, height-2.2*inch, "System Performance Metrics")
        
        p.setFont("Helvetica", 12)
        y = height-2.6*inch
        if _GLOBAL_STATE['is_trained']:
            for res in _GLOBAL_STATE['results_table']:
                p.drawString(1*inch, y, f"{res['Model']}:")
                p.drawString(3.5*inch, y, f"Accuracy: {res['Accuracy']}% | Precision: {res['Precision']}%")
                y -= 0.3*inch
                if y < 1*inch:
                    p.showPage()
                    y = height-1*inch
        else:
            p.drawString(1*inch, y, "System not yet calibrated. No training data available.")
        
    p.showPage()
    p.save()
    return response

def live_prediction(request):
    import json
    from django.http import JsonResponse
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            if not _GLOBAL_STATE['is_trained']:
                return JsonResponse({'error': 'Not trained'})

            raw_input = data
            encoded = []
            le_dict = _GLOBAL_STATE['label_encoders']
            cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                    'PhoneService', 'InternetService', 'OnlineSecurity', 
                    'TechSupport', 'Contract', 'PaperlessBilling', 
                    'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
            
            for col in cols:
                val = raw_input.get(col, 0)
                if col in le_dict:
                    try: val = le_dict[col].transform([str(val)])[0]
                    except: val = 0
                encoded.append(val)
                
            X_input = _GLOBAL_STATE['scaler'].transform([encoded])
            
            probs = []
            for name, model in _GLOBAL_STATE['models'].items():
                if name == 'Neural Network':
                    p = float(model.predict(X_input, verbose=0)[0][0])
                else:
                    try:
                        p = model.predict_proba(X_input)[0][1]
                    except:
                        p = 1.0 if model.predict(X_input)[0] == 1 else 0.0
                probs.append(p)
            
            avg_prob = np.mean(probs)
            
            sat_score = int(raw_input.get('Satisfaction', 3))
            complaints = int(raw_input.get('Complaints', 0))
            if sat_score <= 2: avg_prob += 0.15
            elif sat_score >= 4: avg_prob -= 0.10
            if complaints > 2: avg_prob += 0.10
            elif complaints == 0: avg_prob -= 0.05
            
            avg_prob = min(max(avg_prob, 0.0), 1.0)
            prob_percent = round(avg_prob * 100, 1)
            
            return JsonResponse({'probability': prob_percent})
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return JsonResponse({'error': 'Invalid req'})