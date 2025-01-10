#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install SpeechRecognition pyttsx3 pyaudio


# In[ ]:


import pyttsx3
import speech_recognition as sr  # Corrected the import

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

def speak(text):
    """Function to make the bot speak."""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen to the user's voice command."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("I am listening, please speak.")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)  # Corrected the method name
            return command.lower()
        except sr.UnknownValueError:
            speak("Sorry, I did not catch that.")
            return ""
        except sr.RequestError:
            speak("Could not request results, check your network connection.")
            return ""

def main():
    """Main function to run the voice bot."""
    speak("Hello! I am your basic voice bot from Mallareddy College.")
    while True:
        command = listen()
        if "hello" in command:
            speak("Hi there! Welcome to Mallareddy college.")
        elif "what's your name" in command or "what is your name" in command:
            speak("My name is Basic Bot from Mallareddy College.")
        elif "goodbye" in command:
            speak("Goodbye! Have a great day at Mallareddy college.")
            break  # Exit the loop to stop the bot
        else:
            speak("I didn't understand that. Please try again.")

if __name__ == "__main__":
    main()

