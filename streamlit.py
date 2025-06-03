import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="KnowYourHeartRisk - Ketahuilah Risiko Serangan Jantung Anda",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    try:
        with open("best_model_knn.pkl", "rb") as f:
            knn, accuracy_knn, prec_knn, rec_knn, f1_knn = pickle.load(f)
        return knn, accuracy_knn
    except FileNotFoundError:
        st.error("File model 'best_model_knn.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

knn_model, model_accuracy = load_model()

def preprocess_and_predict(age, gender, cholesterol, heart_rate, exercise_hours, diet,
                           sedentary_hours, weight_kg, height_cm,
                           sleep_hours, systolic_bp, diastolic_bp):
    if knn_model is None:
        return None, None

    if height_cm == 0:
        st.warning("Tinggi badan tidak boleh nol untuk perhitungan BMI.")
        return None, None

    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)

    features = [
        float(age),
        float(gender), 
        float(cholesterol),
        float(heart_rate),
        float(exercise_hours),
        float(diet),
        float(sedentary_hours),
        float(bmi), 
        float(sleep_hours),
        float(systolic_bp),
        float(diastolic_bp),
    ]
    
    try:
        input_array = np.array(features).reshape(1, -1)
        prediction = knn_model.predict(input_array)
        probability = knn_model.predict_proba(input_array) 
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        return None, None

st.markdown("<h1 style='text-align: center; color: #FF6347;'>🫀Prediksi Risiko Serangan Jantung dengan KnowYourHeartRisk!🫀</h1>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
**KnowYourHeartRisk** - Aplikasi ini menggunakan AI untuk memprediksi risiko serangan jantung berdasarkan data kesehatan Anda.
Silakan isi semua field di bawah ini dengan data yang akurat.
""")

if not knn_model and not model_accuracy:
    st.sidebar.error("Model tidak dapat dimuat. Fungsi prediksi tidak akan berjalan.")

input_container = st.sidebar

with input_container:
    st.header("👤 Data Diri & Gaya Hidup")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=50, step=1)
        gender_options = {"Laki-laki": 1, "Perempuan": 0}
        gender = st.selectbox("Jenis Kelamin", options=list(gender_options.keys()), format_func=lambda x: x, index=0)
        gender_val = gender_options[gender]
        exercise_hours = st.number_input("Jam Olahraga per Minggu", min_value=0.0, max_value=50.0, value=3.0, step=0.5)

    with col2:
        weight_kg = st.number_input("Berat Badan (kg)", min_value=10.0, max_value=300.0, value=70.0, step=0.5)
        height_cm = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.5)


    st.markdown("---")
    st.header("🩺 Kondisi Medis & Riwayat")
    col3, col4 = st.columns(2)

    with col3:
        cholesterol = st.number_input("Tingkat Kolesterol (mg/dL)", min_value=50, max_value=600, value=200, step=1)
        heart_rate = st.number_input("Denyut Jantung Istirahat (bpm)", min_value=30, max_value=200, value=70, step=1)

    with col4:
        systolic_bp = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=70, max_value=250, value=120, step=1)
        diastolic_bp = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=40, max_value=150, value=80, step=1)
    st.markdown("---")
    st.header("🔄 Kebiasaan & Faktor Lain")
    col5, col6 = st.columns(2)

    with col5:
        sedentary_hours = st.number_input("Jam Tidak Aktif/Hari (duduk, rebahan)", min_value=0.0, max_value=24.0, value=8.0, step=0.5)
        diet_options = {"Buruk": 0, "Cukup": 1, "Baik": 2, "Sangat Baik": 3}
        diet_selected = st.select_slider("Kualitas Diet Secara Umum", options=list(diet_options.keys()), value="Cukup")
        diet_val = diet_options[diet_selected]

    with col6:
        sleep_hours = st.number_input("Jam Tidur per Hari", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

st.markdown("---")
col_button_1, col_button_2, col_button_3 = st.columns([2,3,2])
with col_button_2:
    predict_button = st.button("🩺 Prediksi Risiko Saya!", use_container_width=True)

if predict_button and knn_model:
    prediction_result, prediction_proba = preprocess_and_predict(
        age, gender_val, cholesterol, heart_rate, exercise_hours, diet_val,
        sedentary_hours, weight_kg, height_cm, sleep_hours, systolic_bp, diastolic_bp
    )


    if prediction_result is not None:
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>Hasil Prediksi</h2>", unsafe_allow_html=True)

        hasil_teks = 'Tinggi' if prediction_result == 1 else 'Rendah'
        prob_tinggi = prediction_proba[1] * 100 
        prob_rendah = prediction_proba[0] * 100 

        if hasil_teks == 'Tinggi':
            st.error(f"**Risiko Serangan Jantung Anda: {hasil_teks}**")
            st.progress(int(prob_tinggi)/100)
            st.markdown(f"<p style='text-align: center;'>Probabilitas Risiko Tinggi: <strong>{prob_tinggi:.2f}%</strong></p>", unsafe_allow_html=True)
        else:
            st.success(f"**Risiko Serangan Jantung Anda: {hasil_teks}**")
            st.progress(int(prob_rendah)/100)
            st.markdown(f"<p style='text-align: center;'>Probabilitas Risiko Rendah: <strong>{prob_rendah:.2f}%</strong></p>", unsafe_allow_html=True)

        height_m_display = height_cm / 100.0
        bmi_display = weight_kg / (height_m_display * height_m_display) if height_m_display > 0 else "N/A"
        st.markdown("---")
        st.subheader("📝 Data Anda")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(f"""
            - **Usia:** {age} tahun  
            - **Jenis Kelamin:** {"Laki-laki" if gender_val == 1 else "Perempuan"}  
            - **Berat Badan:** {weight_kg:.1f} kg  
            - **Tinggi Badan:** {height_cm:.1f} cm  
            - **BMI:** {bmi_display:.2f}  
            - **Kolesterol:** {cholesterol} mg/dL  
            """)

        with col_right:
            st.markdown(f"""
            - **Denyut Jantung:** {heart_rate} bpm  
            - **Tekanan Darah:** {systolic_bp}/{diastolic_bp} mmHg  
            - **Olahraga/Minggu:** {exercise_hours} jam  
            - **Tidak Aktif/Hari:** {sedentary_hours} jam  
            - **Tidur/Hari:** {sleep_hours} jam  
            - **Kualitas Diet:** {diet_selected}
        """)
        st.markdown("---")
        st.subheader("💡 Saran & Analisis untuk Anda")

        penjelasan = []

        if age >= 60:
            penjelasan.append("👉 Usia Anda **{} tahun**, termasuk kategori lanjut. Seiring bertambahnya usia, risiko jantung memang meningkat. Tetap aktif dan pantau kesehatan secara rutin Anda!".format(age))

        if gender_val == 1:
            penjelasan.append("👉 Laki-laki cenderung memiliki risiko lebih tinggi terkena serangan jantung. Tetapi jangan khawatir, pola hidup sehat bisa mengurangi risikonya.")

        if cholesterol >= 240:
            penjelasan.append("⚠️ Kolesterol Anda sangat tinggi (**{} mg/dL**). Cobalah untuk mengurangi makanan berlemak jenuh serta perbanyak konsumsi sayur dan buah untuk mengurangi kolestrol Anda.".format(cholesterol))
        elif cholesterol >= 200:
            penjelasan.append("🟠 Kolesterol Anda agak tinggi (**{} mg/dL**). Sebaiknya mulai jaga pola makan Anda dan rutin periksa kadar kolesterol.".format(cholesterol))

        if heart_rate < 60:
            penjelasan.append("🧘‍♂️ Detak jantung istirahat Anda cukup rendah (**{} bpm**). Jika Anda seorang atlet mungkin ini normal, namun jika tidak, sebaiknya segera lakukan konsultasi ke dokter atau rumah sakit.".format(heart_rate))
        elif heart_rate > 100:
            penjelasan.append("💓 Detak jantung istirahat Anda tinggi (**{} bpm**). Bisa jadi Anda stres, kurang tidur, atau ada faktor lain yang perlu diperiksa. Segera konsultasikan ke dokter atau rumah sakit.".format(heart_rate))

        if exercise_hours < 1:
            penjelasan.append("🏃‍♂️ Anda hanya berolahraga sekitar **{:.1f} jam/minggu**. Ayo mulai rutin berolahraga, minimal 2.5 jam per minggu agar jantung Anda tetap terjaga dan sehat!".format(exercise_hours))

        if diet_val == 0:
            penjelasan.append("🍔 Pola makan Anda terindikasi **kurang baik**. Kurangi makanan cepat saji dan tingkatkan konsumsi asupan buah, sayur, dan serat.")
        elif diet_val == 1:
            penjelasan.append("🥗 Diet Anda masih bisa ditingkatkan. Konsumsi lebih banyak makanan sehat seperti sayur dan buah!")

        if sedentary_hours >= 8:
            penjelasan.append("🪑 Anda menghabiskan sekitar **{:.1f} jam/hari** dengan duduk atau berbaring. Yuk, selingi dengan jalan kaki singkat atau peregangan tiap jam!".format(sedentary_hours))

        if bmi_display >= 30:
            penjelasan.append(f"⚠️ **BMI Anda** ({bmi_display:.1f}) termasuk obesitas. Dengan berat badan {weight_kg:.1f} kg, menurunkan berat sedikit demi sedikit bisa berdampak besar bagi jantung Anda.")
        elif bmi_display >= 25:
            penjelasan.append(f"📉 **BMI Anda** ({bmi_display:.1f}) menunjukkan kelebihan berat badan. Menurunkan beberapa kilogram bisa membantu mengurangi tekanan pada jantung Anda.")

        if sleep_hours < 6:
            penjelasan.append("🌙 Anda tidur kurang dari 6 jam per hari. Tidur yang cukup penting untuk mengatur tekanan darah dan kadar stres bagi jantung Anda.")
        elif sleep_hours > 9:
            penjelasan.append("🛌 Anda tidur lebih dari 9 jam. Tidur cukup itu baik, tapi terlalu lama tidur bisa mengurangi aktivitas Anda. Coba beraktivitas lebih banyak di siang hari!")

        if systolic_bp >= 140 or diastolic_bp >= 90:
            penjelasan.append(f"🩸 Tekanan darah Anda tinggi ({systolic_bp}/{diastolic_bp} mmHg). Hipertensi (tekanan darah tinggi) adalah salah satu pemicu utama serangan jantung. Segera konsultasikan ke dokter dan mulai perbaiki gaya hidup dan pola makan Anda.")

        if penjelasan:
            for saran in penjelasan:
                st.markdown(saran)
        else:
            st.success("🎉 Berdasarkan data Anda, tidak ditemukan faktor risiko yang signifikan. Tetap pertahankan gaya hidup sehat, ya!")

    elif knn_model is None:
        st.warning("Model belum dimuat. Prediksi tidak dapat dilakukan.")


st.markdown("---")