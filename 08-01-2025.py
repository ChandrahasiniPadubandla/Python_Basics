{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4ab4c889-6aed-47be-919b-72916675cdfd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting SpeechRecognition\n",
      "  Using cached SpeechRecognition-3.13.0-py3-none-any.whl.metadata (30 kB)\n",
      "Collecting pyttsx3\n",
      "  Using cached pyttsx3-2.98-py3-none-any.whl.metadata (3.8 kB)\n",
      "Collecting pyaudio\n",
      "  Using cached PyAudio-0.2.14-cp312-cp312-win_amd64.whl.metadata (2.7 kB)\n",
      "Requirement already satisfied: typing-extensions in c:\\users\\indrani\\anaconda3\\lib\\site-packages (from SpeechRecognition) (4.11.0)\n",
      "Collecting comtypes (from pyttsx3)\n",
      "  Using cached comtypes-1.4.9-py3-none-any.whl.metadata (7.1 kB)\n",
      "Collecting pypiwin32 (from pyttsx3)\n",
      "  Using cached pypiwin32-223-py3-none-any.whl.metadata (236 bytes)\n",
      "Requirement already satisfied: pywin32 in c:\\users\\indrani\\anaconda3\\lib\\site-packages (from pyttsx3) (305.1)\n",
      "Using cached SpeechRecognition-3.13.0-py3-none-any.whl (32.8 MB)\n",
      "Using cached pyttsx3-2.98-py3-none-any.whl (34 kB)\n",
      "Using cached PyAudio-0.2.14-cp312-cp312-win_amd64.whl (164 kB)\n",
      "Using cached comtypes-1.4.9-py3-none-any.whl (234 kB)\n",
      "Using cached pypiwin32-223-py3-none-any.whl (1.7 kB)\n",
      "Installing collected packages: pyaudio, SpeechRecognition, pypiwin32, comtypes, pyttsx3\n",
      "Successfully installed SpeechRecognition-3.13.0 comtypes-1.4.9 pyaudio-0.2.14 pypiwin32-223 pyttsx3-2.98\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install SpeechRecognition pyttsx3 pyaudio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "176c5230-e52b-434b-964e-6ec0e6d84825",
   "metadata": {},
   "outputs": [
    {
     "name": "stdin",
     "output_type": "stream",
     "text": [
      "You:  hello\n",
      "You:  goobye\n"
     ]
    }
   ],
   "source": [
    "import pyttsx3\n",
    "\n",
    "engine = pyttsx3.init()\n",
    "\n",
    "def speak(text):\n",
    "  engine.say(text)\n",
    "  engine.runAndWait()\n",
    "\n",
    "def main():\n",
    "  speak(\"Hello! I am your simple bot from Mallareddy University\")\n",
    "  speak(\"You can say hello,ask my name, or say goodbye.\")\n",
    "\n",
    "  while True:\n",
    "    command = input(\"You: \").lower()\n",
    "\n",
    "    if \"hello\" in command:\n",
    "      speak(\"Hello! How can I assist you today?\")\n",
    "\n",
    "    elif \"name\" in command:\n",
    "      speak(\"I am a simple bot. You can call me MalliBot.\")\n",
    "\n",
    "    elif \"HAHA\" in command:\n",
    "      speak(\"haha\")\n",
    "\n",
    "    elif \"goobye\" in command:\n",
    "      speak(\"Goodbye! Have a great day!\")\n",
    "      break\n",
    "    else:\n",
    "      speak(\"I'm sorry, I didn't understand that. Please try again.\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "  main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c42927e-7e1e-44c3-80e6-948f7e505c76",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
