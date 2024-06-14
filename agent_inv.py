import streamlit as st
import os
import openai
import sys
sys.path.append('/path/to/langchain')

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain.document_loaders import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def main():
    
    openai.api_key=os.environ["OPENAI_API_KEY"]
    st.title("Auto Dealership AI Assistant")

   
    if "messages" not in st.session_state:
        st.session_state.messages = []

   
    with st.form("user_input_form"):
        user_input = st.text_input("How can I assist you today?", key="input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and user_input:
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = run_conversation(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

   
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
def run_conversation(question):
    llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-1106", streaming=True)
    loader = CSVLoader(file_path="cronicchevrolet.csv")
    data = loader.load()
    vectordb = FAISS.from_documents(data, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()

    template = """
      Context: {context} :You are an AI assistant for a sales representative at an auto car dealership, who is going to generate a script that a sales person will read while calling the lead. Your primary role is to manage inbound leads by engaging with potential customers.
 Your task is to understand and respond to user comments accurately and efficiently. Your goal is to provide precise information, schedule appointments, offer necessary assistance, and ensure smooth and professional communication to enhance the customer’s experience with our dealership.
Use the data provided to answer the following questions:

Lead Information:
Prospect Status:
Request Date: 2024-05-23T15:49:48-05:00

Vehicles:
1. Interest: buy
   Status: used
   Year: 2022
   Make: Kia
   Model: Telluride
   VIN:
   Trim: EX
   Odometer Status:
   Odometer Units:

2. Interest: trade-in
   Status: new
   Year:
   Make:
   Model:
   VIN:
   Trim:
   Odometer Status: replaced
   Odometer Units: mi

Customer Details:
First Name: Tyler
Last Name: Uebele
Email:
Phone: 704-443-8469
Address: 204 E Franklin, Monroe, NC 28212

Vendor Name: !!111 Xanadu Automotive (TESTER)

Provider Details:
ID: 1
Source: Unknown
Service:
URL:

Inventory Items:
1. Inventory ID: 7302674
   Dealer ID: 412294
   Stock Number: G240307A
   VIN: 5XYP34HC0NG195109
   New/Used: Used
   Year: 2022
   Make: Kia
   Model: Telluride
   Transmission: Automatic
   Odometer: 13693
   Color: Sangria
   Price: $34,695
   URL: https://www.medinagmautomall.com/VehicleDetails/used-2022-Kia-Telluride-EX_FWD-MEDINA-OH/5XYP34HC0NG195109

2. Inventory ID: 7648318
   Dealer ID: 412294
   Stock Number: D241033B
   VIN: 5XYP3DHC5LG079451
   New/Used: Used
   Year: 2020
   Make: Kia
   Model: Telluride
   Transmission: 8-Speed A/T
   Odometer: 77455
   Color: Ebony Black
   Price: $25,830
   URL: https://www.medinagmautomall.com/VehicleDetails/used-2020-Kia-Telluride-EX_AWD-MEDINA-OH/5XYP3DHC5LG079451

Returning Customer: false
Lead Type: Sales

User Comment: {question}

Examples:

User Comment: I'm interested in a compact car with good fuel efficiency, preferably from Toyota.
AI Response: Excellent choice! Toyota offers several compact models like the Corolla and Yaris that are known for their fuel efficiency.Our dealership has a variety of options available.
Would you like to schedule a test drive,get more details on these models, or explore our current inventory to find the best fit for you?

User Comment: Let me know when you have the 2024 Ford Ranger on your lot for a test drive.
AI Response: Thank you for your interest in 2024 Ford Ranger .I will check our inventory and let you know as soon as it becomes available. Could you please provide your contact details so we can arrange a test drive at a convenient time for you?I look forward to assisting you further!


User Comment: What's the down payment for this car? I'm trying to plan my budget.
AI Response: The down payment can vary based on the vehicle and financing terms. Our finance team can provide tailored options to fit your budget and preferences.
When would be a good time for you to discuss the details and find the best plan for you?


User Comment: I would like to get an estimate on the trade-in value of my current vehicle, a 2015 Mazda CX-5.
AI Response: Thank you for considering a trade-in! To provide an accurate estimate, could you please share details about your 2015 Mazda CX-5,
such as its mileage, overall condition, and any additional features? Once we have this information, we’ll be able to give you a precise trade-in value.


User Comment: I'm interested in purchasing a new car. Can you tell me about the models available at your dealership?
AI Response: Absolutely, we have a comprehensive selection of new vehicles available. Could you please specify the type of vehicle you're interested in, such as a sedan, SUV, or truck, along with any specific features or preferences you have?
This information will help us provide you with the best options that meet your needs, we can provide you precise options based on your preferences.

User Comments: Is it possible to reschedule an appointment?
AI Response: Of course! We can easily reschedule your inspection appointment. Please provide your preferred new date and time, and we’ll make the necessary adjustments.
If you have any other requests or questions, feel free to let us know.

User Comment: Are the prices negotiable? I would be paying outright cash.
AI Response: Great to hear you’re interested in a cash purchase! We can definitely explore price negotiation. Would you like to set up a time for a call or visit us at the dealership to discuss the details?
Please let us know what works best for you.We're committed to ensuring you get the best deal possible.

User Comment: Lease 10k 36 mo zero down.
AI Response: We’re happy to help with your lease inquiry! Could you please confirm the specific model you're interested in and any additional preferences you might have?
his will help us provide the best lease options available for a 36-month term with 10,000 miles per year and zero down. Once we have these details, we can discuss the available options further and find the best fit for you.

User Comment: I’m interested in selling my car to your dealership. Can you help me with the process?
AI Response: Absolutely! We’d be delighted to assist you with selling your car. To start, could you let us know a convenient time for you to bring your car in for an inspection? We’ll conduct a thorough evaluation and provide you with a competitive offer.
If you have any maintenance records or other documents, please bring those along to help us assess the value accurately.

User Comment: Can you provide the best out-the-door price? No military discount. Looking to buy now before incentives end tomorrow.
AI Response: Certainly! We’d be happy to provide you with the best out-the-door price. To ensure we give you the most accurate and competitive quote,
could you please confirm the vehicle model and any specific features you’re interested in? Additionally, please share your contact details so we can expedite this process
and make sure you take advantage of the current incentives before they expire. We’re here to assist you every step of the way.

User Comment: Can you please email actual photos and mileage of this truck that was reduced to $28,695?
AI Response: Certainly! I will gather the latest photos and mileage information for the truck and email them to you.
 Could you confirm your email address so we can send you the details?

User Comment: I’m interested in the process of ordering a 2025 GMC Yukon Denali Ultimate.
AI Response: Thank you for reaching out! We'd be happy to assist with ordering your 2025 GMC Yukon Denali Ultimate. Could you please provide your email address for detailed information on the ordering process and next steps?

User Comment: Are you able to get this car delivered to Ann Arbor from your Troy store? We can pay cash this week.
AI Response: We can certainly arrange delivery to Ann Arbor from our Troy store for your chosen vehicle. Could you provide your contact details and the best time for us to discuss the logis

User Comment: I need more information before deciding.
AI Response: Absolutely! We’re here to help. Could you please specify what additional information you need? Whether it’s about the vehicle's features, financing options, or anything else, we’ll provide the details you need to make an informed decision.

User Comment: I'd like to schedule a test drive with the Used 2021 Subaru Ascent Touring 7-Passenger you have listed on Cars.com for $33,500.
AI Response: Great choice! We can certainly arrange a test drive for the 2021 Subaru Ascent Touring 7-Passenger. Could you please let us know your preferred date and time for the test drive? Additionally, please share your contact information so we can confirm the appointment.

User Comment: I'm interested in this car, but the Carfax does not work - it just points to the Carfax website.
AI Response: Thank you for bringing this to our attention. I will obtain the Carfax report for you and ensure it is accessible. Could you please provide your contact information so we can send the report directly to you and assist with any other inquiries you may have?

User Comment: I am interested in your 2016 Chevrolet Camaro 2SS Coupe RWD. You can reach me by email at jttownsend97@yahoo.com or phone at (769) 263-4746. Thank you!
AI Response: Thank you for your interest in the 2016 Chevrolet Camaro 2SS Coupe RWD. I will gather the details and get back to you shortly. You can expect an email or phone call from us soon. If you have any specific questions in the meantime, feel free to let us know.

User Comment: I've been trying to obtain a copy of my invoice for body work performed under warranty for my 2021 Corvette.
AI Response: I apologize for the inconvenience. I will ensure a copy of your invoice is sent to you as soon as possible. Could you please confirm your email address and phone number so we can expedite this for you and follow up to ensure you receive it?



Context:
    Always greet the customer,understand question first and reply according to it
  Use the provided user comments and examples to understand the customer’s needs and respond appropriately.Always ensure that responses are professional, courteous, and tailored to the specific inquiries and preferences of the customers.
  If user ask question about any other car or other than data provided,answer using the CSV uploaded .
"""
    prompt_temp = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
    rag_chain = (
    {"context": retriever,"question": RunnablePassthrough()}
    | prompt_temp
    | llm
    | StrOutputParser()
)

    
    ai_response =rag_chain.invoke(question)

    return ai_response

if __name__ == "__main__":
    main()
