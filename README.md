# Travel Chatbot

An interactive travel planning chatbot that helps users plan their trips with AI-powered recommendations, interactive maps, and detailed itineraries.

## Features

- Interactive chat interface for trip planning
- AI-powered travel recommendations
- Interactive maps with route visualization
- Detailed daily itineraries
- Airport code resolution
- Google Maps integration
- Beautiful UI with Streamlit

## Setup

1. Clone the repository:
```bash
git clone https://github.com/haorui-zhang/travel-chatbot.git
cd travel-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run iternary_chatbot.py
```

## Usage

1. Enter your trip details in the form:
   - Country
   - Departure city
   - Destinations
   - Start date
   - Trip duration

2. Chat with the AI travel advisor to refine your plans

3. Click "Ready to Plan" when you're satisfied with the details

4. View your interactive map and detailed itinerary

## Live Demo

Visit the live demo at: [Your Streamlit Cloud URL]

## Technologies Used

- Python
- Streamlit
- LangChain
- OpenAI GPT-4
- Folium
- Geopy
- Matplotlib

## License

MIT License 