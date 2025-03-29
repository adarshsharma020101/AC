 # agents/viz_agent.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import logging
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Union
import uuid
from datetime import datetime

class VizAgent:
    """
    Agent responsible for creating visualizations from data.
    Supports multiple visualization types and libraries.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.viz_cache = {}
        self.color_palettes = {
            "default": px.colors.qualitative.Plotly,
            "categorical": px.colors.qualitative.Set3,
            "sequential": px.colors.sequential.Blues,
            "diverging": px.colors.diverging.RdBu,
            "monochrome": px.colors.sequential.gray
        }
        
        # Set default Seaborn style
        sns.set(style="whitegrid")
        
        # Create visualization directory if it doesn't exist
        os.makedirs("data/visualizations", exist_ok=True)
    
    def run(self, **params) -> Dict[str, Any]:
        """Main entry point for the visualization agent"""
        viz_type = params.get("viz_type", "auto")
        data = params.get("data")
        
        if data is None:
            raise ValueError("Data parameter is required")
        
        # Allow visualizing based on dataset_id
        if isinstance(data, str):
            # Assuming DataAgent has a get_dataset method
            from agents.data_agent import DataAgent
            data_agent = DataAgent()
            data = data_agent.get_dataset(data)
        
        if viz_type == "auto":
            return self.recommend_visualizations(data, **params)
        else:
            return self.create_visualization(data, viz_type=viz_type, **params)
    
    def create_visualization(self, 
                            data: pd.DataFrame, 
                            viz_type: str,
                            x: Optional[str] = None,
                            y: Optional[Union[str, List[str]]] = None,
                            color: Optional[str] = None,
                            size: Optional[str] = None,
                            facet: Optional[str] = None,
                            title: Optional[str] = None,
                            palette: Optional[str] = "default",
                            library: str = "plotly",
                            interactive: bool = True,
                            height: int = 500,
                            width: int = 800,
                            **kwargs) -> Dict[str, Any]:
        """
        Create a visualization based on parameters
        
        Args:
            data: DataFrame containing the data
            viz_type: Type of visualization (bar, line, scatter, etc.)
            x: Column name for x-axis
            y: Column name(s) for y-axis
            color: Column name for color encoding
            size: Column name for size encoding
            facet: Column name for faceting
            title: Chart title
            palette: Color palette name
            library: Visualization library to use (plotly, matplotlib, seaborn)
            interactive: Whether to create an interactive plot
            height: Plot height in pixels
            width: Plot width in pixels
            
        Returns:
            Dictionary with visualization details and data
        """
        # Generate a unique ID for this visualization
        viz_id = str(uuid.uuid4())
        
        # Default title if not provided
        if not title:
            if x and y:
                if isinstance(y, list):
                    title = f"{', '.join(y)} by {x}"
                else:
                    title = f"{y} by {x}"
            else:
                title = f"{viz_type.capitalize()} Chart"
        
        # Select color palette
        colors = self.color_palettes.get(palette, self.color_palettes["default"])
        
        # Create visualization based on type and library
        try:
            if library == "plotly":
                fig = self._create_plotly_viz(data, viz_type, x, y, color, size, facet, title, colors, **kwargs)
                
                # Update figure layout
                fig.update_layout(
                    title=title,
                    height=height, 
                    width=width,
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                
                # Convert to JSON for storage
                plot_json = fig.to_json()
                
                # Save HTML file
                html_path = f"data/visualizations/{viz_id}.html"
                fig.write_html(html_path)
                
                # Create a lightweight representation for preview
                fig_preview = fig.to_image(format="png", width=width, height=height)
                preview_b64 = base64.b64encode(fig_preview).decode("utf-8")
                
            elif library == "matplotlib" or library == "seaborn":
                fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
                
                if viz_type == "bar":
                    if isinstance(y, list):
                        df_melted = data.melt(id_vars=x, value_vars=y, var_name='variable', value_name='value')
                        sns.barplot(data=df_melted, x=x, y='value', hue='variable', ax=ax, palette=colors)
                    else:
                        sns.barplot(data=data, x=x, y=y, hue=color, ax=ax, palette=colors)
                
                elif viz_type == "line":
                    if isinstance(y, list):
                        for column in y:
                            ax.plot(data[x], data[column], label=column)
                        ax.legend()
                    else:
                        sns.lineplot(data=data, x=x, y=y, hue=color, ax=ax, palette=colors)
                
                elif viz_type == "scatter":
                    sns.scatterplot(data=data, x=x, y=y, hue=color, size=size, ax=ax, palette=colors)
                
                elif viz_type == "histogram":
                    sns.histplot(data=data, x=x, hue=color, ax=ax, palette=colors)
                
                elif viz_type == "boxplot":
                    sns.boxplot(data=data, x=x, y=y, hue=color, ax=ax, palette=colors)
                
                elif viz_type == "heatmap":
                    # Create pivot table if x and y are provided
                    if x and y and color:
                        pivot_data = data.pivot_table(index=y, columns=x, values=color, aggfunc='mean')
                        sns.heatmap(pivot_data, annot=kwargs.get("annot", True), cmap=colors, ax=ax)
                    else:
                        # Correlation heatmap if no specific columns
                        corr_data = data.select_dtypes(include=[np.number]).corr()
                        sns.heatmap(corr_data, annot=kwargs.get("annot", True), cmap=colors, ax=ax)
                
                elif viz_type == "pie":
                    if x and not y:
                        # Count values in x
                        counts = data[x].value_counts()
                        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors)
                    elif x and y:
                        # Use y values with x as labels
                        ax.pie(data[y], labels=data[x], autopct='%1.1f%%', colors=colors)
                
                else:
                    raise ValueError(f"Unsupported visualization type: {viz_type} for {library}")
                
                # Set title and labels
                ax.set_title(title)
                if x:
                    ax.set_xlabel(x)
                if y and not isinstance(y, list):
                    ax.set_ylabel(y)
                
                # Save to buffer and convert to base64
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)
                
                # Save to file
                img_path = f"data/visualizations/{viz_id}.png"
                fig.savefig(img_path, dpi=100, bbox_inches="tight")
                
                # There's no direct JSON representation for matplotlib
                plot_json = json.dumps({
                    "type": "matplotlib",
                    "viz_type": viz_type,
                    "parameters": {
                        "x": x,
                        "y": y,
                        "color": color,
                        "size": size,
                        "title": title
                    }
                })
                
                preview_b64 = img_data
            
            else:
                raise ValueError(f"Unsupported visualization library: {library}")
            
            # Store visualization metadata
            viz_metadata = {
                "viz_id": viz_id,
                "viz_type": viz_type,
                "created_at": datetime.now().isoformat(),
                "parameters": {
                    "x": x,
                    "y": y,
                    "color": color,
                    "size": size,
                    "facet": facet,
                    "title": title,
                    "palette": palette,
                    "library": library,
                    "interactive": interactive,
                    "height": height,
                    "width": width
                },
                "columns_used": [col for col in [x, y, color, size, facet] if col],
                "data_shape": data.shape
            }
            
            # Cache the visualization
            self.viz_cache[viz_id] = viz_metadata
            
            # Save metadata
            with open(f"data/visualizations/{viz_id}_metadata.json", "w") as f:
                json.dump(viz_metadata, f, indent=2)
            
            return {
                "success": True,
                "viz_id": viz_id,
                "title": title,
                "viz_type": viz_type,
                "plot_data": plot_json,
                "preview": f"data:image/png;base64,{preview_b64}",
                "file_path": f"data/visualizations/{viz_id}.{'html' if library == 'plotly' else 'png'}",
                "metadata": viz_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "viz_type": viz_type,
                "parameters": {
                    "x": x,
                    "y": y,
                    "color": color,
                    "size": size
                }
            }
    
    def recommend_visualizations(self, data: pd.DataFrame, max_recommendations: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Recommend appropriate visualizations based on data characteristics
        
        Args:
            data: DataFrame to visualize
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            Dictionary with recommended visualizations
        """
        recommendations = []
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_columns = data.select_dtypes(include=["datetime"]).columns.tolist()
        
        # Rule-based recommendations
        
        # 1. Time series visualizations if datetime columns exist
        if datetime_columns:
            time_col = datetime_columns[0]
            
            for num_col in numeric_columns[:3]:  # Limit to first 3 numeric columns
                recommendations.append({
                    "viz_type": "line",
                    "x": time_col,
                    "y": num_col,
                    "title": f"{num_col} over Time",
                    "description": f"Time series showing how {num_col} changes over time"
                })
        
        # 2. Bar charts for categorical variables
        for cat_col in categorical_columns[:2]:  # Limit to first 2 categorical columns
            value_counts = data[cat_col].value_counts()
            if 2 <= len(value_counts) <= 15:  # Ideal for bar charts
                recommendations.append({
                    "viz_type": "bar",
                    "x": cat_col,
                    "title": f"Distribution of {cat_col}",
                    "description": f"Bar chart showing the distribution of {cat_col} categories"
                })
                
                # If we have numeric columns, suggest aggregations
                if numeric_columns:
                    for num_col in numeric_columns[:2]:
                        recommendations.append({
                            "viz_type": "bar",
                            "x": cat_col,
                            "y": num_col,
                            "title": f"Average {num_col} by {cat_col}",
                            "description": f"Bar chart showing average {num_col} for each {cat_col} category"
                        })
        
        # 3. Histograms for numeric variables
        for num_col in numeric_columns[:3]:
            recommendations.append({
                "viz_type": "histogram",
                "x": num_col,
                "title": f"Distribution of {num_col}",
                "description": f"Histogram showing the distribution of {num_col} values"
            })
        
        # 4. Scatter plots for pairs of numeric variables
        if len(numeric_columns) >= 2:
            for i, x_col in enumerate(numeric_columns[:3]):
                for y_col in numeric_columns[i+1:min(i+3, len(numeric_columns))]:
                    recommendations.append({
                        "viz_type": "scatter",
                        "x": x_col,
                        "y": y_col,
                        "title": f"{y_col} vs {x_col}",
                        "description": f"Scatter plot showing relationship between {x_col} and {y_col}"
                    })
                    
                    # Add color by categorical if available
                    if categorical_columns:
                        recommendations.append({
                            "viz_type": "scatter",
                            "x": x_col,
                            "y": y_col,
                            "color": categorical_columns[0],
                            "title": f"{y_col} vs {x_col} by {categorical_columns[0]}",
                            "description": f"Scatter plot with points colored by {categorical_columns[0]}"
                        })
        
        # 5. Correlation heatmap for numeric variables
        if len(numeric_columns) >= 3:
            recommendations.append({
                "viz_type": "heatmap",
                "title": "Correlation Heatmap",
                "description": "Heatmap showing correlations between numeric variables"
            })
        
        # 6. Box plots for numeric variables by categories
        if categorical_columns and numeric_columns:
            for cat_col in categorical_columns[:1]:
                for num_col in numeric_columns[:3]:
                    recommendations.append({
                        "viz_type": "boxplot",
                        "x": cat_col,
                        "y": num_col,
                        "title": f"Distribution of {num_col} by {cat_col}",
                        "description": f"Box plot showing how {num_col} is distributed across {cat_col} categories"
                    })
        
        # Limit to max_recommendations
        recommendations = recommendations[:max_recommendations]
        
        # Create a preview for each recommendation
        for i, rec in enumerate(recommendations):
            try:
                # Create a small preview version
                preview = self.create_visualization(
                    data=data,
                    viz_type=rec["viz_type"],
                    x=rec.get("x"),
                    y=rec.get("y"),
                    color=rec.get("color"),
                    title=rec.get("title"),
                    height=300,
                    width=400
                )
                
                if preview["success"]:
                    recommendations[i]["preview_id"] = preview["viz_id"]
                    recommendations[i]["preview_image"] = preview["preview"]
            except Exception as e:
                self.logger.error(f"Error creating preview for recommendation {i}: {str(e)}")
        
        return {
            "success": True,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    def get_visualization(self, viz_id: str) -> Dict[str, Any]:
        """Retrieve a visualization by ID"""
        if viz_id not in self.viz_cache:
            # Try to load from disk
            metadata_path = f"data/visualizations/{viz_id}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    self.viz_cache[viz_id] = json.load(f)
            else:
                raise ValueError(f"Visualization with ID {viz_id} not found")
        
        viz_metadata = self.viz_cache[viz_id]
        
        # Determine file path based on library
        if viz_metadata["parameters"]["library"] == "plotly":
            file_path = f"data/visualizations/{viz_id}.html"
        else:
            file_path = f"data/visualizations/{viz_id}.png"
        
        # Read file
        if os.path.exists(file_path):
            if file_path.endswith(".png"):
                with open(file_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                preview = f"data:image/png;base64,{image_data}"
            else:
                with open(file_path, "r") as f:
                    html_content = f.read()
                preview = None  # HTML content can't be directly previewed
        else:
            preview = None
        
        return {
            "success": True,
            "viz_id": viz_id,
            "metadata": viz_metadata,
            "file_path": file_path,
            "preview": preview
        }
    
    def list_visualizations(self) -> List[Dict[str, Any]]:
        """List all available visualizations with basic metadata"""
        # Refresh cache from disk
        viz_files = [f for f in os.listdir("data/visualizations") if f.endswith("_metadata.json")]
        
        for viz_file in viz_files:
            viz_id = viz_file.replace("_metadata.json", "")
            if viz_id not in self.viz_cache:
                try:
                    with open(f"data/visualizations/{viz_file}", "r") as f:
                        self.viz_cache[viz_id] = json.load(f)
                except:
                    pass
        
        return [{
            "viz_id": viz_id,
            "title": metadata.get("parameters", {}).get("title", f"Visualization {viz_id}"),
            "viz_type": metadata.get("viz_type", "unknown"),
            "created_at": metadata.get("created_at", ""),
            "library": metadata.get("parameters", {}).get("library", "unknown")
        } for viz_id, metadata in self.viz_cache.items()]
    
    def _create_plotly_viz(self, 
                          data: pd.DataFrame, 
                          viz_type: str,
                          x: Optional[str] = None,
                          y: Optional[Union[str, List[str]]] = None,
                          color: Optional[str] = None,
                          size: Optional[str] = None,
                          facet: Optional[str] = None,
                          title: Optional[str] = None,
                          colors: Optional[List] = None,
                          **kwargs) -> go.Figure:
        """Create a Plotly visualization based on parameters"""
        
        if viz_type == "bar":
            if isinstance(y, list):
                # Multiple bars per x value
                fig = go.Figure()
                for y_col in y:
                    fig.add_trace(go.Bar(x=data[x], y=data[y_col], name=y_col))
            elif y is not None:
                # Standard bar chart
                fig = px.bar(data, x=x, y=y, color=color, title=title, color_discrete_sequence=colors)
            else:
                # Count plot
                value_counts = data[x].value_counts().reset_index()
                value_counts.columns = [x, 'count']
                fig = px.bar(value_counts, x=x, y='count', title=title, color_discrete_sequence=colors)
        
        elif viz_type == "line":
            if isinstance(y, list):
                # Multiple lines
                fig = go.Figure()
                for y_col in y:
                    fig.add_trace(go.Scatter(x=data[x], y=data[y_col], mode='lines', name=y_col))
            else:
                # Standard line chart
                fig = px.line(data, x=x, y=y, color=color, title=title, color_discrete_sequence=colors)
        
        elif viz_type == "scatter":
            fig = px.scatter(data, x=x, y=y, color=color, size=size, facet_col=facet, 
                             title=title, color_discrete_sequence=colors)
        
        elif viz_type == "histogram":
            fig = px.histogram(data, x=x, color=color, title=title, color_discrete_sequence=colors)
        
        elif viz_type == "boxplot":
            fig = px.box(data, x=x, y=y, color=color, title=title, color_discrete_sequence=colors)
        
        elif viz_type == "violin":
            fig = px.violin(data, x=x, y=y, color=color, title=title, color_discrete_sequence=colors)
        
        elif viz_type == "heatmap":
            if x and y and color:
                # Create pivot table
                pivot_data = data.pivot_table(index=y, columns=x, values=color, aggfunc='mean')
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale=colors
                ))
            else:
                # Correlation heatmap
                corr_data = data.select_dtypes(include=[np.number]).corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.index,
                    colorscale=colors
                ))
        
        elif viz_type == "pie":
            if x and not y:
                # Count values in x
                counts = data[x].value_counts().reset_index()
                counts.columns = [x, 'count']
                fig = px.pie(counts, names=x, values='count', title=title, color_discrete_sequence=colors)
            elif x and y:
                # Use y values with x as labels
                fig = px.pie(data, names=x, values=y, title=title, color_discrete_sequence=colors)
        
        elif viz_type == "area":
            if isinstance(y, list):
                # Multiple area charts
                fig = go.Figure()
                for y_col in y:
                    fig.add_trace(go.Scatter(x=data[x], y=data[y_col], mode='lines', fill='tozeroy', name=y_col))
            else:
                # Standard area chart
                fig = px.area(data, x=x, y=y, color=color, title=title, color_discrete_sequence=colors)
        
        elif viz_type == "density_heatmap":
            fig = px.density_heatmap(data, x=x, y=y, title=title, color_continuous_scale=colors)
        
        # elif viz_type == "treemap":
        #     fig = px.treemap(data, path=[x] if isinstance(x, str) else x, values=y, color=color, 
        #                     title=title, color_discrete_sequence=colors
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type} for Plotly")
