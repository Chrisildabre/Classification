import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf=pickle.load(open('clf.pkl', 'rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))

def cleanresume(resume_text):
  cleantext=re.sub("http\S+\s"," ",resume_text)#remove urls
  cleantext=re.sub("#\S+\s"," ",cleantext)#hashtags
  cleantext=re.sub("@\S+"," ",cleantext)#remove mentions
  #punctuations and special characters
  cleantext=re.sub(r'[^\x00-\x7f]'," ",cleantext)
  cleantext=re.sub('\s+'," ",cleantext)
  cleantext=re.sub('RT|CC'," ",cleantext)
  cleantext=re.sub('[%s]'% re.escape("""!"#$%'><.,?/[/]@&*(/){|}\~`:;-+_'=""")," ",cleantext)
  return cleantext

#Web App
def main():
    st.title("Resume Screening app")
    Uploadfile=st.file_uploader('Upload resume', type=['txt','pdf'])

    if Uploadfile is not None:
        try:
            resume_file=Uploadfile.read()
            resume_text=resume_file.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_file.decode('latin-1')


        cleaned_resume=cleanresume(resume_text)
        input_features=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]
        st.write(prediction_id)


        category_mapping={
            15:"Java Developer",
            23:"Testing",
            8:"Devops Engineer",
            20:"python Developer",
            24:"Web Designing",
            17:"Network security engineer",
            0:"Advocate",
            5:"Civil Engineer",
            21:"SAP Developer",
            12:"HR",
            13:"Hadoop",
            3:"Blockchain",
            10:"ETL Developer",
            18:"Operations Manager",
            6:"Data Science",
            22:"Sales",
            16:"Mechanical Engineer",
            1:"Arts",
            7:"Database Engineer",
            11:"Electical Engineer",
            14:"Health and Fitness",
            19:"PMO",
            4:"Business Analyst",
            9:"Dotnet Developer",
            2:"Automation Testing",
        }

        Category_name=category_mapping.get(prediction_id,"Unknown")
        st.write("Prediction Category: ",Category_name)

        

if __name__ == '__main__':
    main()