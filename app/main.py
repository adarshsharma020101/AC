 # app/main.py

import streamlit as st
import os
import sys
import logging
from datetime import datetime

# Add project root to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components
from app.components.sidebar import render_sidebar
from app.components.data_upload import render_data_upload
from app.components.visualization import render_visualization
from app.components.voice_interface import render_voice_interface
from app.components.collaborative import render_collaborative_tools
from app.components.anomaly_detection import render_anomaly_detection

# Import agents
from agents.agent_manager import AgentManager
from agents.data_agent import DataAgent
from agents.report_agent import ReportAgent
from agents.chat_agent import ChatAgent
from agents.viz_agent import VizAgent
from agents.forecast_agent import ForecastAgent
from agents.anomaly_agent import AnomalyAgent
from agents.ethics_agent import EthicsAgent

# Import utils
from app.utils.session_manager import initialize_session, get_session_state

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def initialize_agents():
    """Initialize and register all agents with the agent manager"""
    agent_manager = AgentManager()
    
    # Initialize individual agents
    data_agent = DataAgent()
    report_agent = ReportAgent()
    chat_agent = ChatAgent()
    viz_agent = VizAgent()
    forecast_agent = ForecastAgent()
    anomaly_agent = AnomalyAgent()
    ethics_agent = EthicsAgent()
    
    # Register agents with manager
    agent_manager.register_agent("data", data_agent)
    agent_manager.register_agent("report", report_agent)
    agent_manager.register_agent("chat", chat_agent)
    agent_manager.register_agent("viz", viz_agent)
    agent_manager.register_agent("forecast", forecast_agent)
    agent_manager.register_agent("anomaly", anomaly_agent)
    agent_manager.register_agent("ethics", ethics_agent)
    
    return agent_manager

def main():
    """Main entry point for the Streamlit application"""
    st.set_page_config(
        page_title="Business Data AI Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state if not already done
    initialize_session()
    session = get_session_state()
    
    # Initialize agents if not already done
    if 'agent_manager' not in session:
        session['agent_manager'] = initialize_agents()
        logger.info("Agent manager initialized")
    
    # Page title and description
    st.title("Business Data AI Tool")
    st.write("""
    Convert your raw business data into interactive reports, presentations, 
    and chatbot-driven summaries for better decision-making.
    """)
    
    # Render sidebar for navigation
    selected_page = render_sidebar()
    
    # Main content area based on selected page
    if selected_page == "Upload":
        render_data_upload()
    elif selected_page == "Anomaly Detection":  
        render_anomaly_detection()
    elif selected_page == "Chat":
        render_chat_interface()
    elif selected_page == "Visualize":
        render_visualization()
    elif selected_page == "Reports":
        render_reports_interface()
    elif selected_page == "Collaborate":
        render_collaborative_tools()
    elif selected_page == "Voice":
        render_voice_interface()
    elif selected_page == "Settings":
        render_settings()
    
    
    # Footer
    st.markdown("---")
    st.markdown(f"© {datetime.now().year} Business Data AI Tool. All rights reserved.")
    
    # Collect user feedback for continuous improvement
    if st.button("Submit Feedback"):
        collect_user_feedback()

def render_chat_interface():
    """Render the chat interface for interacting with the chat agent"""
    st.header("Chat with Your Data")
    
    # Get chat agent from the agent manager
    session = get_session_state()
    agent_manager = session['agent_manager']
    chat_agent = agent_manager.get_agent("chat")
    
    # Initialize chat history if not already present
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Display chat history
    for message in session['chat_history']:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask something about your data...")
    if user_input:
        # Add user message to chat history
        session['chat_history'].append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process with chat agent
        with st.spinner("Thinking..."):
            try:
                # Call the chat agent
                response = chat_agent.process_query(
                    query=user_input, 
                    history=session['chat_history']
                )
                
                # Add agent response to chat history
                session['chat_history'].append({"role": "assistant", "content": response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response)
            except Exception as e:
                st.error(f"Error processing your query: {str(e)}")

def render_reports_interface():
    """Render the reports generation interface"""
    st.header("Generate Reports & Presentations")
    
    # Get report agent from the agent manager
    session = get_session_state()
    agent_manager = session['agent_manager']
    report_agent = agent_manager.get_agent("report")
    
    # Report configuration options
    report_type = st.selectbox(
        "Report Type", 
        ["Executive Summary", "Detailed Analysis", "Presentation", "Dashboard"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        include_visuals = st.checkbox("Include Visualizations", value=True)
        include_forecasts = st.checkbox("Include Forecasts", value=False)
    with col2:
        theme = st.selectbox("Theme", ["Professional", "Minimalist", "Colorful", "Custom"])
        audience = st.selectbox("Target Audience", ["Executives", "Managers", "Analysts", "Stakeholders"])
    
    # Additional options based on report type
    if report_type == "Presentation":
        slides_count = st.slider("Number of Slides", 5, 30, 15)
    elif report_type == "Dashboard":
        layout = st.selectbox("Layout", ["Grid", "Flowing", "Tabbed"])
    
    # Generate report button
    if st.button("Generate Report"):
        with st.spinner("Generating your report..."):
            try:
                # Prepare parameters for the report agent
                params = {
                    "report_type": report_type,
                    "include_visuals": include_visuals,
                    "include_forecasts": include_forecasts,
                    "theme": theme,
                    "audience": audience
                }
                
                # Add type-specific parameters
                if report_type == "Presentation":
                    params["slides_count"] = slides_count
                elif report_type == "Dashboard":
                    params["layout"] = layout
                
                # Generate the report
                report = report_agent.generate_report(**params)
                
                # Display or download the report
                st.success("Report generated successfully!")
                st.download_button(
                    label="Download Report",
                    data=report["content"],
                    file_name=report["filename"],
                    mime=report["mime_type"]
                )
                
                # Preview if applicable
                if "preview" in report:
                    st.subheader("Preview")
                    st.write(report["preview"])
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

def render_settings():
    """Render the settings page"""
    st.header("Settings")
    
    # User settings
    st.subheader("User Settings")
    st.text_input("Username", value="Admin")
    st.text_input("Email", value="admin@example.com")
    
    # Agent settings
    st.subheader("Agent Settings")
    agents = get_session_state()['agent_manager'].list_agents()
    for agent in agents:
        st.checkbox(f"Enable {agent.capitalize()} Agent", value=True)
    
    # Model settings
    st.subheader("Model Settings")
    st.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Hugging Face", "Local"])
    st.slider("Temperature", 0.0, 1.0, 0.7)
    st.slider("Response Max Length", 100, 2000, 500)
    
    # Data settings
    st.subheader("Data Settings")
    st.checkbox("Enable Data Versioning", value=True)
    st.checkbox("Generate Synthetic Data for Testing", value=False)
    
    # Save settings button
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")

def render_anomaly_detection():
    """Render the anomaly detection interface"""
    st.header("Anomaly Detection Center")
    
    session = get_session_state()
    agent_manager = session['agent_manager']
    anomaly_agent = agent_manager.get_agent("anomaly")
    data_agent = agent_manager.get_agent("data")
    
    # Dataset selection
    datasets = data_agent.list_datasets()
    if not datasets:
        st.warning("Please upload and process data first")
        return
        
    selected_dataset = st.selectbox(
        "Select Dataset",
        [d['dataset_id'] for d in datasets],
        help="Choose a processed dataset for anomaly detection"
    )
    
    # Detection configuration
    with st.expander("Advanced Settings"):
        methods = st.multiselect(
            "Detection Methods",
            ["Autoencoder", "Isolation Forest", "Prophet", "Z-Score"],
            default=["Autoencoder", "Isolation Forest"]
        )
        
        consensus_mode = st.selectbox(
            "Consensus Strategy",
            ["Ensemble", "AOM", "MOA"],
            index=0
        )
    
    if st.button("Run Anomaly Scan"):
        with st.spinner("Detecting anomalies..."):
            try:
                result = anomaly_agent.detect_anomalies(
                    dataset_id=selected_dataset,
                    methods=methods,
                    mode=consensus_mode.lower()
                )
                
                session['last_anomaly_result'] = result
                st.success(f"Found {sum(result['report']['anomaly_flags']} anomalies")
                
            except Exception as e:
                st.error(f"Detection failed: {str(e)}")
    
    # Display results if available
    if 'last_anomaly_result' in session:
        result = session['last_anomaly_result']
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Anomalies", sum(result['report']['anomaly_flags']))
        with col2:
            st.metric("Max Severity", f"{max(result['report']['scores']):.2f}")
        with col3:
            st.metric("Detection Methods", ", ".join(result['metadata']['methods_used']))
        
        # Show visualizations
        st.subheader("Detection Visualizations")
        viz_agent = agent_manager.get_agent("viz")
        
        for viz_type, viz_data in result['visualizations'].items():
            if viz_data:
                with st.expander(f"{viz_type.replace('_', ' ').title()}"):
                    if viz_data['file_path'].endswith('.html'):
                        st.components.v1.html(open(viz_data['file_path']).read(), 
                                           height=400)
                    else:
                        st.image(viz_data['preview'], 
                               caption=viz_data['metadata']['parameters']['title'])
        
        # Anomaly explanation and feedback
        st.subheader("Anomaly Investigation")
        selected_anomaly = st.selectbox(
            "Select Anomaly to Investigate",
            [i for i, flag in enumerate(result['report']['anomaly_flags']) if flag]
        )
        
        if selected_anomaly:
            explanation = anomaly_agent.explain_anomalies(
                dataset_id=selected_dataset,
                version_id=result['version_id'],
                anomaly_index=selected_anomaly
            )
            
            st.write("### Feature Contributions")
            for feature in explanation['local_explanations']:
                st.progress(abs(feature['shap_impact']), 
                          text=f"{feature['feature']}: {feature['value']}")
            
            # Feedback system
            st.write("### Feedback")
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                if st.button("👍 Confirm Anomaly"):
                    anomaly_agent.process_feedback({
                        'version_id': result['version_id'],
                        'indexes': [selected_anomaly],
                        'label': True
                    })
                    st.success("Feedback recorded!")
            with col_fb2:
                if st.button("👎 False Positive"):
                    anomaly_agent.process_feedback({
                        'version_id': result['version_id'],
                        'indexes': [selected_anomaly],
                        'label': False
                    })
                    st.success("Feedback recorded!")

def collect_user_feedback():
    """Collect and store user feedback for model improvement"""
    st.subheader("Submit Feedback")
    
    feedback_type = st.selectbox(
        "Feedback Type", 
        ["Bug Report", "Feature Request", "Usability Issue", "General Feedback"]
    )
    
    feedback_text = st.text_area("Your Feedback")
    rating = st.slider("Rate your experience", 1, 5, 3)
    
    if st.button("Submit"):
        if feedback_text:
            # Store feedback in the data/feedback directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            feedback_data = {
                "type": feedback_type,
                "text": feedback_text,
                "rating": rating,
                "timestamp": timestamp
            }
            
            # In a real app, save this to a file or database
            st.success("Thank you for your feedback! It will help us improve the system.")
            
            # You would typically save to a file or database here
            # with open(f"data/feedback/feedback_{timestamp}.json", "w") as f:
            #     json.dump(feedback_data, f)
        else:
            st.warning("Please enter your feedback before submitting.")

if __name__ == "__main__":
    main()
