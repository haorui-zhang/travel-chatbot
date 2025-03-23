# Interactive Travel Chatbot

An intelligent travel planning assistant that helps users create personalized travel itineraries using AI. The chatbot provides interactive planning, route optimization, and visual trip mapping.

## Features

- 🤖 Interactive chat interface for travel planning
- 🌍 Multi-destination trip planning
- 🗺️ Interactive map visualization with route planning
- 📅 Detailed daily itineraries
- ✈️ Airport code resolution
- 🎯 Route optimization
- 📱 Beautiful Streamlit UI

## Prerequisites

- Python 3.8+
- OpenAI API key
- Internet connection for map services

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/travel-chatbot.git
cd travel-chatbot
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
```bash
streamlit run iternary_chatbot.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Start planning your trip by:
   - Entering your departure city
   - Selecting destinations
   - Specifying trip duration
   - Choosing travel dates

## Features in Detail

- **Interactive Chat**: Natural conversation with the AI travel advisor
- **Route Optimization**: Smart planning of travel routes between destinations
- **Visual Maps**: Interactive Folium maps showing your travel route
- **Detailed Itineraries**: Day-by-day breakdown of activities and travel plans
- **Google Maps Integration**: Direct links to view routes on Google Maps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 API
- Streamlit for the web interface
- Folium for map visualization
- Geopy for geocoding services 