import langchain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import re
import urllib.parse
import folium
from folium.features import DivIcon, CustomIcon
from geopy.geocoders import Nominatim
import webbrowser
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import numpy as np
from urllib.parse import quote
from dotenv import load_dotenv
import streamlit.components.v1 as components
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Initialize OpenAI client with API key from environment
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# STEP 0: Chat interaction to validate user preferences
def chat_with_user(initial_input: str = None) -> dict:
    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        if initial_input:
            st.session_state.messages.append({"role": "user", "content": initial_input})
    
    if 'ready_to_plan' not in st.session_state:
        st.session_state.ready_to_plan = False

    # Display chat history
    st.markdown("### 💬 Chat with your Travel Advisor")
    st.markdown("---")
    
    # Create main chat container
    chat_container = st.container()
    
    # Display messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Initial AI response if this is the first message
        if len(st.session_state.messages) == 1:
            chat_prompt = ChatPromptTemplate.from_template("""
            You are a friendly travel advisor. Based on the user's input, ask relevant follow-up questions to ensure we have all necessary details and validate their travel plans. Consider:

            1. If the destinations are realistic given the duration
            2. If the time of year is suitable for their chosen destinations
            3. If there might be better alternatives for their interests
            4. Any potential travel restrictions or visa requirements
            5. Special considerations based on their departure city

            Current input: {user_input}

            Provide your response in a conversational way, asking at most 2 questions at a time.
            Also mention that the user can click the 'Ready to Plan' button when they want to proceed with the trip planning.
            """)
            
            chat_chain = chat_prompt | llm
            with st.chat_message("assistant"):
                response = chat_chain.invoke({"user_input": st.session_state.messages[0]["content"]})
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

    # Add some spacing before the Ready to Plan button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not st.session_state.ready_to_plan:
        # Add the "Ready to Plan" button
        if len(st.session_state.messages) >= 2:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🎯 I'm Ready to Plan My Trip", key="ready_button", type="primary", use_container_width=True):
                    st.session_state.ready_to_plan = True
                    st.rerun()
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate AI response
            with st.chat_message("assistant"):
                # Prepare full context from chat history
                full_context = "\n".join(msg["content"] for msg in st.session_state.messages)
                
                chat_prompt = ChatPromptTemplate.from_template("""
                You are a friendly travel advisor. Based on the conversation history, continue helping the user plan their trip.
                Continue asking relevant questions (maximum 2 at a time) about their travel preferences.
                Remember to mention that they can click the 'Ready to Plan' button whenever they feel ready to proceed with the detailed trip planning.

                Previous conversation:
                {user_input}

                Respond naturally and conversationally.
                """)
                
                chat_chain = chat_prompt | llm
                with st.spinner("Thinking..."):
                    response = chat_chain.invoke({"user_input": full_context})
                    st.markdown(response.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.content})

    # If ready to plan, return the chat history and final context
    if st.session_state.ready_to_plan:
        full_context = "\n".join(msg["content"] for msg in st.session_state.messages)
        return {
            "chat_history": st.session_state.messages,
            "final_input": full_context
        }
    
    return None

# STEP 0: Streamlit interactive UI

def streamlit_user_input():
    st.title("✈️ Interactive Travel Chatbot")
    st.markdown("""
    Welcome to your personal travel planning assistant! Let's plan your perfect trip together.
    Fill out the form below to get started.
    """)
    
    # Initialize session state for form values
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    if 'initial_prompt' not in st.session_state:
        st.session_state.initial_prompt = None
    if 'form_values' not in st.session_state:
        st.session_state.form_values = {
            'country': "Brazil",
            'departure_city': "New York City",
            'destinations': "Rio de Janeiro, São Paulo"
        }
    
    # Initial form for basic information
    if not st.session_state.form_submitted:
        with st.form("trip_form", clear_on_submit=False):
            st.subheader("🌍 Trip Details")
            col1, col2 = st.columns(2)
            
            with col1:
                country = st.text_input(
                    "Which country are you traveling to?",
                    value=st.session_state.form_values['country']
                )
                start_date = st.date_input("Start date of your trip")
                duration_days = st.number_input("Trip duration (in days)", min_value=1, value=10)
            
            with col2:
                departure_city = st.text_input(
                    "Where are you departing from?",
                    value=st.session_state.form_values['departure_city']
                )
                destinations = st.text_area(
                    "Destinations you want to visit (comma-separated)",
                    value=st.session_state.form_values['destinations']
                )
            
            submit = st.form_submit_button("Start Planning", use_container_width=True)

        if submit:
            st.session_state.form_submitted = True
            # Save form values
            st.session_state.form_values = {
                'country': country,
                'departure_city': departure_city,
                'destinations': destinations
            }
            st.session_state.initial_prompt = (
                f"I want to go to {country} in {start_date.strftime('%B')} for {duration_days} days, starting on {start_date.strftime('%Y-%m-%d')}, "
                f"from {departure_city}. I want to visit {destinations}."
            )
            # Clear any existing chat history when starting new conversation
            if 'messages' in st.session_state:
                del st.session_state.messages
            if 'ready_to_plan' in st.session_state:
                del st.session_state.ready_to_plan
            # Initialize messages with the initial prompt
            st.session_state.messages = [{"role": "user", "content": st.session_state.initial_prompt}]
            st.rerun()  # Rerun after form submission to start fresh
    
    # Continue chat if form was submitted
    if st.session_state.form_submitted:
        chat_result = chat_with_user(None)  # Pass None since we already added the initial prompt to messages
        if chat_result is not None:
            return chat_result["final_input"]
            
    return None

# STEP 1: Extract structured travel plan
response_schemas = [
    ResponseSchema(name="country", description="Country of travel"),
    ResponseSchema(name="cities", description="List of cities/places to visit EXCLUDING the departure city"),
    ResponseSchema(name="month", description="Month of travel (optional if exact start date is given)"),
    ResponseSchema(name="duration_days", description="Trip duration in days"),
    ResponseSchema(name="departure_city", description="User's city of departure (should be preserved exactly as provided)"),
    ResponseSchema(name="start_date", description="Exact starting date of the trip (e.g. 2025-04-05), if mentioned"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

travel_prompt = PromptTemplate(
    template="""Extract the structured travel plan from the user's message. If locations are vague (e.g. 'bossa nova city', 'sand desert in the north'), infer the most likely city name (e.g. 'Rio de Janeiro', 'São Luís') and return city-level locations.

IMPORTANT: 
1. The departure city MUST be exactly '{departure_city}' - do not modify or infer anything else
2. The 'cities' list should ONLY include the destinations to visit, NOT the departure city
3. Preserve the exact departure city in the output

{format_instructions}

User: {user_input}
""",
    input_variables=["user_input", "departure_city"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

travel_chain = travel_prompt | llm | parser

# STEP 2: GPT function to resolve location → airport code
airport_prompt = ChatPromptTemplate.from_template(
    "Given the city '{location}', what is the nearest major international airport in {country}?\n"
    "Only return the IATA airport code and city name in this format: 'SLZ - São Luís'."
)
airport_chain = airport_prompt | llm

def resolve_airport(location: str, country: str = "Brazil") -> str:
    try:
        result = airport_chain.invoke({"location": location, "country": country})
        return result.content.strip()
    except Exception as e:
        return f"❌ Error resolving {location}: {e}"

# STEP 3: Route Optimization Prompt
route_prompt = PromptTemplate(
    template="""You are a travel route optimizer. Given the following trip details, create an optimized travel plan that includes:
1. A route order that minimizes travel time and maximizes sightseeing
2. Day allocation for each destination
3. A daily itinerary with specific activities

IMPORTANT: The trip MUST start and end in {origin}. Always include flights from and back to {origin} in the itinerary.

Trip Details:
- Starting from: {origin}
- Country: {country}
- Destinations: {destinations}
- Duration: {days} days
- Start Date: {start_date}

Format your response as follows:
### Route Order:
- [{origin} → City1 → City2 → ... → {origin}]

### Day Allocation:
- City1: X days
- City2: Y days
(Include travel days in the allocation)

### Daily Itinerary:
- Day 1: Departure from {origin}, arrival in [First City]. [Activities]
...
- Day {days}: Morning activities in [Last City], afternoon/evening flight back to {origin}

Remember to:
1. Include realistic travel times between cities
2. Account for time zones and jet lag
3. Include specific activities and sights for each day
4. Ensure the last day includes the return flight to {origin}
""",
    input_variables=["origin", "country", "destinations", "days", "start_date"]
)

route_chain = route_prompt | llm

def optimize_route(origin: str, country: str, destinations: list, days: int, start_date: str):
    try:
        # Ensure origin is included in the route planning
        response = route_chain.invoke({
            "origin": origin,
            "country": country,
            "destinations": ", ".join(destinations),
            "days": days,
            "start_date": start_date or "unknown"
        })
        print("\n🧠 Optimized Travel Plan:")
        print(response.content.strip())
        return response.content.strip()
    except Exception as e:
        return f"❌ Error optimizing route: {e}"

# STEP 4: Extract route cities and day allocation from GPT output
def extract_city_route_from_gpt_output(text):
    """Extract the route from GPT's response."""
    try:
        # First try to find the explicit route section
        route_section = re.search(r'### Route Order:.*?-.*?\[(.*?)\]', text, re.DOTALL)
        if route_section:
            # Extract cities from the route
            route = route_section.group(1).strip()
            cities = [city.strip() for city in route.split('→')]
            
            # Filter out empty strings and get only the destination cities
            filtered_cities = []
            for i, city in enumerate(cities):
                if city and city.strip():
                    # Skip if it's the first or last city (departure city)
                    if i != 0 and i != len(cities) - 1:
                        filtered_cities.append(city.strip())
            
            if filtered_cities:
                return filtered_cities
        
        # If no route section found, try to extract from day allocations
        allocations = extract_day_allocations(text)
        if allocations:
            # Filter out travel days and get destination cities
            destinations = [city for city, days in allocations.items() 
                          if not city.lower().startswith(('travel', 'departure', 'arrival'))]
            if destinations:
                return destinations
            
    except Exception as e:
        print(f"Error extracting route: {e}")
    return None

def extract_day_allocations(gpt_text: str):
    """Extract day allocations from the GPT response."""
    allocations = {}
    
    # Try to find explicit day allocations section
    allocation_section = re.search(r'### Day Allocation:(.*?)(?=###|$)', gpt_text, re.DOTALL)
    if allocation_section:
        text = allocation_section.group(1)
        # Extract city and days, excluding Buffer/Extra days
        for match in re.findall(r'[-•]\s*(.*?):\s*(\d+)\s*days?', text, re.IGNORECASE):
            city, days = match
            city = city.strip()
            # Skip if the city contains 'buffer' or 'extra' (case insensitive)
            if not any(word in city.lower() for word in ['buffer', 'extra', 'travel']):
                allocations[city] = int(days)
    
    # If no explicit allocations found, try to extract from daily itinerary
    if not allocations:
        current_city = None
        city_days = {}
        
        # Look for day entries in the itinerary
        for line in gpt_text.split('\n'):
            # Check for city headers
            city_header = re.match(r'(?:[-•]\s*)?([^:]+):', line)
            if city_header:
                current_city = city_header.group(1).strip()
                # Skip if the city contains 'buffer' or 'extra' (case insensitive)
                if not any(word in current_city.lower() for word in ['buffer', 'extra', 'travel']):
                    city_days[current_city] = city_days.get(current_city, 0) + 1
            
            # Check for day entries
            day_entry = re.match(r'.*Day\s+\d+.*', line)
            if day_entry and current_city:
                # Skip if the current city contains 'buffer' or 'extra' (case insensitive)
                if not any(word in current_city.lower() for word in ['buffer', 'extra', 'travel']):
                    city_days[current_city] = city_days.get(current_city, 0) + 1
        
        # Convert to allocations
        for city, days in city_days.items():
            if not any(word in city.lower() for word in ['buffer', 'extra', 'travel']):
                allocations[city] = days
    
    return allocations

def extract_daily_itinerary(gpt_text: str):
    """Extract daily itinerary from the GPT response."""
    itinerary = {}
    
    # Look for the daily itinerary section
    itinerary_section = re.search(r'### Daily Itinerary:(.*?)(?=###|$)', gpt_text, re.DOTALL)
    if itinerary_section:
        text = itinerary_section.group(1).strip()
        
        # Get start date directly from session state
        start_date = None
        if 'form_submitted' in st.session_state and st.session_state.form_submitted:
            try:
                # Convert the date_input value to datetime
                form_date = st.session_state.initial_prompt.split('starting on ')[1].split(',')[0]
                start_date = datetime.strptime(form_date, '%Y-%m-%d')
                print(f"Using start date from form: {start_date}")
            except Exception as e:
                print(f"Error parsing form date: {e}")
        
        # Extract cities from day allocations
        cities = set()
        allocation_section = re.search(r'### Day Allocation:(.*?)(?=###|$)', gpt_text, re.DOTALL)
        if allocation_section:
            for line in allocation_section.group(1).split('\n'):
                if line.strip().startswith('-'):
                    city_match = re.match(r'-\s*(.*?):\s*\d+\s*days?', line.strip())
                    if city_match:
                        city = city_match.group(1).strip()
                        if not any(word in city.lower() for word in ['travel', 'buffer', 'extra']):
                            cities.add(city)
        
        # If no cities found in allocation section, try to extract from route order
        if not cities:
            route_section = re.search(r'### Route Order:.*?-.*?\[(.*?)\]', gpt_text, re.DOTALL)
            if route_section:
                route = route_section.group(1).strip()
                cities.update([city.strip() for city in route.split('→') if city.strip()])
        
        # Split into day blocks
        day_blocks = re.split(r'(?=\*\*Day \d+:)', text)
        
        for block in day_blocks:
            if not block.strip():
                continue
            
            # Extract day number and content
            day_match = re.search(r'\*\*Day (\d+):(.*?)(?=\*\*Day|\Z)', block, re.DOTALL)
            if day_match:
                day_num = int(day_match.group(1))
                day_content = day_match.group(2).strip()
                
                # Calculate the actual date if start_date is available
                if start_date:
                    current_date = start_date + timedelta(days=day_num - 1)
                    date_str = current_date.strftime('%B %d, %Y')  # Format: March 23, 2025
                    print(f"Processing day {day_num}, date: {date_str}")  # Debug print
                else:
                    date_str = f"Day {day_num}"
                    print(f"No start date found, using: {date_str}")  # Debug print
                
                # Find which city this day belongs to
                for city in cities:
                    if city.lower() in day_content.lower():
                        if city not in itinerary:
                            itinerary[city] = []
                        
                        # Clean up the content and remove time markers and asterisks
                        activities = []
                        for line in day_content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('**'):
                                # Remove time markers, bullet points, and asterisks
                                line = re.sub(r'^[-•]\s*', '', line)
                                line = re.sub(r'(Morning|Afternoon|Evening|Night):\s*', '', line)
                                line = re.sub(r'\*\*', '', line)  # Remove asterisks
                                if line.strip():  # Only add non-empty lines
                                    activities.append(line.strip())
                        
                        # Combine all activities into one line
                        if activities:
                            combined_activity = '; '.join(activities)
                            if combined_activity:  # Only add if there are actual activities
                                itinerary[city].append(f"{date_str}: {combined_activity}")
                                print(f"Added activity for {city}: {date_str}")  # Debug print
                        break
    
    return itinerary

# STEP 5: Generate Google Maps route URL
def generate_google_maps_route(cities):
    """Generate a Google Maps URL for the route."""
    if not cities:
        return None
    
    # Create the URL-encoded city list
    encoded_cities = [quote(city.strip()) for city in cities if city.strip()]
    if encoded_cities:
        return f"https://www.google.com/maps/dir/{'/'.join(encoded_cities)}"
    return None

# STEP 6: Plot on folium map with arrows and icons
def plot_route_on_map(cities, allocations, start_date, itinerary):
    """Plot the route on a Folium map with enhanced UI."""
    try:
        # Create a map centered on Brazil
        m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
        
        # Initialize lists for coordinates and city names
        coordinates = []
        city_names = []
        
        # Get coordinates for each city
        for city in cities:
            try:
                location = Nominatim(user_agent="my_agent", timeout=10).geocode(city)
                if location:
                    coordinates.append([location.latitude, location.longitude])
                    city_names.append(city)
                    
                    # Create an enhanced popup for each city
                    days = allocations.get(city, "")
                    days_text = f"{days} days" if days else ""
                    activities = itinerary.get(city, [])
                    
                    # Enhanced popup with detailed itinerary
                    popup_html = f"""
                    <div style="font-family: Arial, sans-serif; max-width: 400px; padding: 15px;">
                        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h3 style="color: #2c3e50; margin-bottom: 10px; border-bottom: 2px solid #3498db; padding-bottom: 5px;">
                                {city}
                            </h3>
                            <div style="color: #e67e22; font-weight: bold; margin-bottom: 15px; font-size: 1.1em;">
                                Duration: {days_text}
                            </div>
                            <div style="margin-top: 15px;">
                                <h4 style="color: #2c3e50; margin-bottom: 10px;">Daily Itinerary:</h4>
                    """
                    
                    # Add activities if available
                    if activities:
                        popup_html += '<div style="margin-left: 10px;">'
                        for activity in activities:
                            # Extract date and content
                            date_content = activity.split(':', 1)
                            if len(date_content) == 2:
                                date = date_content[0].strip()
                                activity_content = date_content[1].strip()
                                popup_html += f"""
                                    <div style="margin-bottom: 10px; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                                        <div style="color: #2c3e50; font-weight: bold; margin-bottom: 5px;">
                                            {date}
                                        </div>
                                        <div style="color: #34495e;">
                                            {activity_content}
                                        </div>
                                    </div>
                                """
                        popup_html += '</div>'
                    else:
                        popup_html += """
                            <div style="color: #7f8c8d; font-style: italic;">
                                No specific activities planned for this location.
                            </div>
                        """
                    
                    popup_html += '</div></div></div>'
                    
                    # Create a marker with custom icon
                    icon = folium.Icon(
                        color='red',
                        icon='info-sign',
                        prefix='fa'
                    )
                    
                    folium.Marker(
                        [location.latitude, location.longitude],
                        popup=folium.Popup(popup_html, max_width=450),
                        icon=icon,
                        tooltip=f"{city} ({days_text})"
                    ).add_to(m)
            except Exception as e:
                print(f"Could not geocode {city}: {e}")
                continue
        
        # Draw flight paths between cities if we have at least 2 locations
        if len(coordinates) >= 2:
            for i in range(len(coordinates) - 1):
                # Draw a line for the flight path
                points = [coordinates[i], coordinates[i + 1]]
                folium.PolyLine(
                    points,
                    weight=2,
                    color='#3498db',
                    opacity=0.8,
                    dash_array='10'
                ).add_to(m)
                
                # Calculate midpoint for flight icon
                mid_lat = (coordinates[i][0] + coordinates[i + 1][0]) / 2
                mid_lon = (coordinates[i][1] + coordinates[i + 1][1]) / 2
                
                # Create a smaller flight icon using DivIcon
                flight_icon_html = """
                <div style="font-size: 12px; color: #3498db;">
                    <i class="fa fa-plane"></i>
                </div>
                """
                flight_icon = folium.DivIcon(
                    html=flight_icon_html,
                    icon_size=(12, 12),
                    icon_anchor=(6, 6)
                )
                
                # Add flight icon with enhanced popup
                flight_popup = f"""
                <div style="font-family: Arial, sans-serif; padding: 10px;">
                    <div style="font-weight: bold; color: #2c3e50; margin-bottom: 5px;">
                        ✈️ {city_names[i]} → {city_names[i + 1]}
                    </div>
                </div>
                """
                
                folium.Marker(
                    [mid_lat, mid_lon],
                    popup=folium.Popup(flight_popup, max_width=250),
                    icon=flight_icon,
                    tooltip=f"Flight: {city_names[i]} → {city_names[i + 1]}"
                ).add_to(m)
        
        return m
    except Exception as e:
        print(f"Error creating map: {e}")
        return None

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the bearing between two points."""
    import math
    
    # Convert to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Calculate bearing
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.degrees(math.atan2(y, x))
    
    # Convert to 0-360
    bearing = (bearing + 360) % 360
    return bearing

def chatbot_travel_planner(user_input):
    print(f"🧠 Interpreting input: {user_input}")

    # Extract original departure city from session state
    departure_city = st.session_state.form_values.get('departure_city', "New York City")
    
    trip = travel_chain.invoke({
        "user_input": user_input,
        "departure_city": departure_city
    })

    print("\n📋 Trip Details:")
    print(f"  Country: {trip['country']}")
    print(f"  Duration: {trip['duration_days']} days")
    print(f"  Month: {trip['month']}")
    print(f"  Start Date: {trip['start_date']}")
    print(f"  Departure City: {trip['departure_city']}")
    print(f"  Destinations: {', '.join(trip['cities'])}")

    print("\n🛬 Airport Mapping:")
    resolved_destinations = []
    # First resolve departure city
    departure_resolved = resolve_airport(departure_city, country="United States")
    print(f"  • {departure_city} → {departure_resolved}")
    resolved_destinations.append(departure_city)
    
    # Then resolve destinations
    for place in trip['cities']:
        resolved = resolve_airport(place, country=trip['country'])
        print(f"  • {place} → {resolved}")
        resolved_destinations.append(place)
    
    # Add departure city at the end to complete the loop
    resolved_destinations.append(departure_city)

    result_text = optimize_route(
        departure_city,  # Use the original departure city
        trip['country'],
        trip['cities'],  # Only pass the destinations, not departure city
        trip['duration_days'],
        trip.get('start_date') or ""
    )

    route = extract_city_route_from_gpt_output(result_text)
    allocations = extract_day_allocations(result_text)
    itinerary = extract_daily_itinerary(result_text)

    if route:
        # Ensure the route starts and ends with departure city
        full_route = [departure_city] + route + [departure_city]
        maps_url = generate_google_maps_route(full_route)
        print("\n🗺️ Google Maps Route:")
        print(maps_url)
        st.markdown(f"[View on Google Maps]({maps_url})")

        print("\n📍 Generating interactive map...")
        m = plot_route_on_map(full_route, allocations, trip['start_date'], itinerary)
        if m is not None:
            m.save("trip_route_map.html")
            try:
                # Display the map directly in Streamlit
                with open("trip_route_map.html", "r", encoding="utf-8") as f:
                    map_html = f.read()
                components.html(map_html, height=600)
            except Exception as e:
                st.warning(f"Could not display map: {str(e)}")
        else:
            st.warning("Could not generate interactive map. Please use the Google Maps link above.")
    else:
        print("⚠️ Could not extract city route from GPT response.")
        st.error("Could not extract city route from GPT response.")

    # Display the itinerary in Streamlit
    st.markdown("## 📅 Your Itinerary")
    for city, days in allocations.items():
        st.markdown(f"### {city} ({days} days)")
        if city in itinerary:
            for activity in itinerary[city]:
                st.markdown(f"- {activity}")

    return resolved_destinations

# Run the app
if __name__ == "__main__":
    try:
        # Configure Streamlit page
        st.set_page_config(
            page_title="Travel Planner",
            page_icon="✈️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add CSS to improve the look
        st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        .main {
            padding: 2rem;
        }
        .chat-message {
            margin-bottom: 1rem;
        }
        .stForm {
            background-color: #f8f9fa;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

        # Add a reset button in the sidebar
        with st.sidebar:
            st.title("🧭 Navigation")
            
            # Add Help Section
            st.markdown("### 📖 How to Use")
            with st.expander("Click to see instructions", expanded=False):
                st.markdown("""
                1. **Fill out the trip form** with:
                   - Country you want to visit
                   - Start date
                   - Trip duration
                   - Departure city
                   - Destinations (comma-separated)

                2. **Chat with the AI** to:
                   - Get travel recommendations
                   - Discuss specific interests
                   - Ask about local events
                   - Get visa information

                3. **Click 'Ready to Plan'** when you want to:
                   - Generate detailed itinerary
                   - See interactive map
                   - Get day-by-day activities

                4. **Start New Trip** to plan another adventure!
                """)

            # Add Quick Overview Section
            st.markdown("### 🌟 Popular Destinations")
            with st.expander("Click to see destinations", expanded=False):
                st.markdown("""
                **Popular Countries:**
                - Brazil (Beaches, Culture)
                - Japan (Technology, Tradition)
                - France (Art, Cuisine)
                - Italy (History, Food)
                - Thailand (Beaches, Temples)

                **Best Time to Visit:**
                - Summer: June-August
                - Winter: December-February
                - Shoulder Season: March-May, September-November
                """)

            # Add Tips Section (folded)
            st.markdown("### 💡 Travel Tips")
            with st.expander("Click to see tips", expanded=False):
                st.markdown("""
                **Before You Go:**
                - Check visa requirements
                - Get travel insurance
                - Learn basic local language
                - Pack for the season

                **During Your Trip:**
                - Use safe transport options
                - Keep valuables secure
                - Drink bottled water
                - Try local cuisine
                """)

            # Add the reset button at the bottom
            st.markdown("---")
            if st.button("Start New Trip", use_container_width=True):
                # Reset all session state
                session_vars = [
                    'form_submitted',
                    'initial_prompt',
                    'messages',
                    'ready_to_plan',
                    'form_values'
                ]
                for key in session_vars:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Main content
        user_input = streamlit_user_input()
        
        # Only proceed with planning if we have input and user clicked ready
        if user_input and st.session_state.get('ready_to_plan', False):
            with st.spinner("Planning your trip..."):
                chatbot_travel_planner(user_input)
                # Reset states after planning is complete
                st.session_state.form_submitted = False
                st.session_state.initial_prompt = None
                st.session_state.ready_to_plan = False
                if 'form_values' in st.session_state:
                    del st.session_state.form_values  # Reset form values for new trip

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try refreshing the page. If the error persists, check the console for more details.")
        raise e