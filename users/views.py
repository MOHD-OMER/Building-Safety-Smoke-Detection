from django.shortcuts import render
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
from sklearn.svm import SVC
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, f1_score
import joblib
import os
import pandas as pd


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.password = make_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid)
            password_valid = check_password(pswd, check.password)
            if not password_valid and not check.password.startswith('pbkdf2_'):
                if check.password == pswd:
                    check.password = make_password(pswd)
                    check.save(update_fields=['password'])
                    password_valid = True
            if not password_valid:
                raise UserRegistrationModel.DoesNotExist
            status = check.status
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your account is not activated yet.')
                return render(request, 'UserLogin.html')
        except UserRegistrationModel.DoesNotExist:
            messages.success(request, 'Invalid login id or password')
    return render(request, 'UserLogin.html', {})

def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = os.path.join(settings.MEDIA_ROOT, 'smoke_detection_iot.csv')
    per_page = 100
    page = max(1, int(request.GET.get('page', 1)))

    if not os.path.exists(path):
        context = {
            'data': '<p class="theme-page-meta">Dataset file not found. Place <strong>smoke_detection_iot.csv</strong> in the <strong>media</strong> folder.</p>',
            'page': 1,
            'num_pages': 1,
            'total_rows': 0,
            'per_page': per_page,
            'start_row': 0,
            'end_row': 0,
        }
        return render(request, 'users/viewdataset.html', context)

    # Get total row count without loading full file
    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        total_rows = sum(1 for _ in f)
    num_pages = max(1, (total_rows + per_page - 1) // per_page)
    page = min(page, num_pages)

    # Read only the current page of rows
    if page == 1:
        df_page = pd.read_csv(path, nrows=per_page + 1)  # header + first per_page data rows
    else:
        skip = range(1, 1 + (page - 1) * per_page)
        df_page = pd.read_csv(path, skiprows=skip, nrows=per_page, header=0)
    df_html = df_page.to_html(classes='table table-striped', index=False)

    context = {
        'data': df_html,
        'page': page,
        'num_pages': num_pages,
        'total_rows': total_rows,
        'per_page': per_page,
        'start_row': (page - 1) * per_page + 1,
        'end_row': min(page * per_page, total_rows),
    }
    return render(request, 'users/viewdataset.html', context)

def Training(request):
    # Function to calculate IoU (Intersection over Union) for binary classification
    def calculate_iou(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        return intersection / union if union != 0 else 0

    # Load dataset
    data_path = os.path.join(settings.MEDIA_ROOT, 'smoke_detection_iot.csv')
    if not os.path.exists(data_path):
        return render(request, 'users/training.html', {
            'results': None,
            'models_info': [],
            'metrics': {},
            'best_model': None,
            'best_auc': 0,
        })
    data = pd.read_csv(data_path)

    # Define features (X) and target (y)
    X = data.drop(['Fire Alarm','S.No','UTC'], axis=1)  # Make sure your target column is named correctly
    y = data['Fire Alarm']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }

    metrics = {}
    best_model = None
    best_auc = 0
    # Orange/smoke palette for UI (order: RandomForest, GradientBoosting, AdaBoost, LogisticRegression, SVM, DecisionTree, KNN)
    model_colors = ['#ff7722', '#a0b0c4', '#607890', '#ff9944', '#ff8800', '#9aaabb', '#576878']

    models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for idx, (name, clf) in enumerate(classifiers.items()):
        print(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        iou = calculate_iou(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics[name] = {
            'Precision': precision,
            'Recall': recall,
            'AUC-ROC': auc,
            'IoU': iou,
            'accuracy': accuracy,
            'f1': f1,
        }

        joblib.dump(clf, os.path.join(models_dir, f'{name}.pkl'))

        if auc > best_auc:
            best_auc = auc
            best_model = name

    # Build results sorted by AUC (best first) for template cards/charts/table
    name_list = list(metrics.keys())
    results = []
    models_info = []
    for i, name in enumerate(name_list):
        m = metrics[name]
        acc_pct = m['accuracy'] * 100
        prec_pct = m['Precision'] * 100
        rec_pct = m['Recall'] * 100
        f1_pct = m['f1'] * 100
        color = model_colors[i % len(model_colors)]
        results.append({
            'name': name,
            'accuracy': acc_pct,
            'precision': prec_pct,
            'recall': rec_pct,
            'f1': f1_pct,
            'color': color,
        })
        models_info.append({'name': name, 'accuracy': acc_pct, 'color': color})
    # Sort by AUC descending so best model is first
    results = sorted(results, key=lambda r: metrics[r['name']]['AUC-ROC'], reverse=True)
    models_info = sorted(models_info, key=lambda m: metrics[m['name']]['AUC-ROC'], reverse=True)

    context = {
        'metrics': metrics,
        'best_model': best_model,
        'best_auc': best_auc,
        'results': results,
        'models_info': models_info,
    }

    return render(request, 'users/training.html', context)

# views.py

from django.conf import settings
import os
import joblib
import pandas as pd
from django.contrib import messages
from django.shortcuts import render

# Reasonable min/max for prediction input validation (sensor ranges)
FEATURE_LIMITS = {
    'Temperature[C]': (-40, 80),
    'Humidity[%]': (0, 100),
    'TVOC[ppb]': (0, 60000),
    'eCO2[ppm]': (400, 8192),
    'Raw H2': (0, 50000),
    'Raw Ethanol': (0, 50000),
    'Pressure[hPa]': (300, 1100),
    'PM1.0': (0, 1000),
    'PM2.5': (0, 1000),
    'NC0.5': (0, 5000),
    'NC1.0': (0, 5000),
    'NC2.5': (0, 2000),
    'CNT': (0, 25000),
}


def Prediction(request):
    FEATURES = [
        'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]',
        'Raw H2', 'Raw Ethanol', 'Pressure[hPa]',
        'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT'
    ]

    result = None

    if request.method == 'POST':
        try:
            input_data = []
            for feature in FEATURES:
                value = request.POST.get(feature)
                if value is None or value == '':
                    messages.error(request, f'Missing input for {feature}')
                    return render(request, 'users/predict_form.html')
                try:
                    num = float(value)
                except (ValueError, TypeError):
                    messages.error(request, f'Invalid number for {feature}. Enter a numeric value.')
                    return render(request, 'users/predict_form.html')
                low, high = FEATURE_LIMITS.get(feature, (None, None))
                if low is not None and (num < low or num > high):
                    messages.error(request, f'{feature} must be between {low} and {high}.')
                    return render(request, 'users/predict_form.html')
                input_data.append(num)

            scaler_path = os.path.join(settings.MEDIA_ROOT, 'scaler.pkl')
            scaler = joblib.load(scaler_path)

            models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
            classifiers = {}
            for model_name in ['RandomForest', 'GradientBoosting', 'AdaBoost', 'LogisticRegression', 'SVM', 'DecisionTree', 'KNN']:
                model_path = os.path.join(models_dir, f'{model_name}.pkl')
                if os.path.exists(model_path):
                    classifiers[model_name] = joblib.load(model_path)

            selected_model = request.POST.get('model', 'RandomForest')
            if selected_model not in classifiers:
                selected_model = 'RandomForest'
            best_model = classifiers[selected_model]

            input_df = pd.DataFrame([input_data], columns=FEATURES)
            input_scaled = scaler.transform(input_df)
            prediction = best_model.predict(input_scaled)

            if isinstance(prediction, (list, np.ndarray)):
                prediction = prediction[0]

            confidence = None
            if hasattr(best_model, 'predict_proba'):
                proba = best_model.predict_proba(input_scaled)[0]
                confidence = float(proba[1] * 100) if len(proba) > 1 else float(proba[0] * 100)

            result = "Fire Alarm: Yes Smoke detected in Building" if prediction == 1 else "Fire Alarm: Smoke Not detected in Building"
        except Exception as e:
            messages.error(request, f"Prediction error: {str(e)}")
            result = None
            prediction = None
            confidence = None
            selected_model = None

    context = {'result': result}
    if request.method == 'POST' and result:
        context['prediction'] = prediction
        context['confidence'] = confidence
        context['model_name'] = selected_model
    return render(request, 'users/predict_form.html', context)

# ================================================================
# PASTE THIS ENTIRE BLOCK AT THE BOTTOM OF users/views.py
# (Replaces any existing CNNPrediction function)
# ================================================================
import json
import uuid
from PIL import Image as PILImage
import io

def CNNPrediction(request):
    result           = None
    confidence       = None
    label            = None
    yolo_img_url     = None
    yolo_detections  = []
    yolo_available   = False

    if request.method == 'POST':
        img_file = request.FILES.get('image')

        if not img_file:
            messages.error(request, 'Please upload an image file.')
            return render(request, 'users/cnn_predict.html', {})

        try:
            import tensorflow as tf

            # ── Save uploaded image to media/uploads/ ────────────────
            uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            ext         = os.path.splitext(img_file.name)[-1].lower() or '.jpg'
            unique_name = f"{uuid.uuid4().hex}{ext}"
            upload_path = os.path.join(uploads_dir, unique_name)

            img_bytes = img_file.read()
            with open(upload_path, 'wb') as f:
                f.write(img_bytes)

            # ══════════════════════════════════════════════════════════
            # 1.  CNN CLASSIFICATION  (original logic — unchanged)
            # ══════════════════════════════════════════════════════════
            model_path   = os.path.join(settings.MEDIA_ROOT, 'cnn_model.h5')
            classes_path = os.path.join(settings.MEDIA_ROOT, 'cnn_classes.json')

            if not os.path.exists(model_path):
                messages.error(request, 'CNN model not found. Run train_cnn.py first.')
                return render(request, 'users/cnn_predict.html', {})

            cnn_model = tf.keras.models.load_model(model_path)

            with open(classes_path, 'r') as f:
                class_indices = json.load(f)
            # {'fire': 0, 'no_fire': 1}  →  sigmoid near 0 = fire

            img_pil     = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
            img_resized = img_pil.resize((224, 224))
            img_array   = np.array(img_resized, dtype=np.float32) / 255.0
            img_array   = np.expand_dims(img_array, axis=0)

            raw_output  = cnn_model.predict(img_array, verbose=0)
            sigmoid_val = float(raw_output[0][0])

            fire_idx  = class_indices.get('fire', 0)
            fire_prob = (1.0 - sigmoid_val) if fire_idx == 0 else sigmoid_val

            if fire_prob >= 0.5:
                result     = 'SMOKE / FIRE DETECTED'
                label      = 'danger'
                confidence = round(fire_prob * 100, 1)
            else:
                result     = 'NO SMOKE DETECTED'
                label      = 'safe'
                confidence = round((1.0 - fire_prob) * 100, 1)

            # ══════════════════════════════════════════════════════════
            # 2.  YOLO OBJECT DETECTION
            # ══════════════════════════════════════════════════════════
            yolo_weights = os.path.join(settings.MEDIA_ROOT, 'models', 'best.pt')

            if os.path.exists(yolo_weights):
                yolo_available = True
                try:
                    from ultralytics import YOLO
                    from PIL import ImageDraw, ImageFont

                    yolo_model   = YOLO(yolo_weights)
                    yolo_results = yolo_model.predict(
                        source=upload_path,
                        conf=0.25,
                        verbose=False
                    )

                    # ── Draw bounding boxes using Pillow (no cv2 needed) ──
                    draw_img = PILImage.open(io.BytesIO(img_bytes)).convert('RGB')
                    draw     = ImageDraw.Draw(draw_img)
                    W, H     = draw_img.size

                    CLASS_COLORS = {
                        'fire':  '#FF5032',
                        'smoke': '#50C8C8',
                    }
                    DEFAULT_COLOR = '#FFA500'

                    for r in yolo_results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf_score      = float(box.conf[0])
                            cls_id          = int(box.cls[0])
                            cls_name        = r.names.get(cls_id, str(cls_id)).lower()

                            color     = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
                            label_txt = f"{cls_name.capitalize()}  {conf_score:.2f}"

                            # Draw box (3px thick)
                            for t in range(3):
                                draw.rectangle(
                                    [x1-t, y1-t, x2+t, y2+t],
                                    outline=color
                                )

                            # Draw label background + text
                            font_size = max(14, int(W / 60))
                            try:
                                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                            except:
                                font = ImageFont.load_default()

                            bbox = draw.textbbox((0, 0), label_txt, font=font)
                            tw   = bbox[2] - bbox[0]
                            th   = bbox[3] - bbox[1]
                            ly1  = max(y1 - th - 8, 0)

                            draw.rectangle([x1, ly1, x1 + tw + 8, ly1 + th + 8], fill=color)
                            draw.text((x1 + 4, ly1 + 2), label_txt, fill='white', font=font)

                            yolo_detections.append({
                                'class': cls_name.capitalize(),
                                'conf':  round(conf_score, 2),
                            })

                    # ── Save annotated image ──────────────────────────────
                    results_dir  = os.path.join(settings.MEDIA_ROOT, 'results')
                    os.makedirs(results_dir, exist_ok=True)
                    out_filename = f"yolo_{unique_name}"
                    out_path     = os.path.join(results_dir, out_filename)
                    draw_img.save(out_path)

                    yolo_img_url = settings.MEDIA_URL + 'results/' + out_filename

                except Exception as yolo_err:
                    import traceback
                    yolo_available  = False
                    yolo_img_url    = None
                    yolo_detections = []
                    print(f"YOLO ERROR FULL: {traceback.format_exc()}")
                    messages.warning(request, f'YOLO detection unavailable: {str(yolo_err)}')
        except Exception as e:
            messages.error(request, f'Prediction error: {str(e)}')
            return render(request, 'users/cnn_predict.html', {})

    context = {
        'result':          result,
        'confidence':      confidence,
        'label':           label,
        'yolo_img_url':    yolo_img_url,
        'yolo_detections': yolo_detections,
        'yolo_available':  yolo_available,
    }
    return render(request, 'users/cnn_predict.html', context)            