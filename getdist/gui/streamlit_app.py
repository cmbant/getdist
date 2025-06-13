import copy
import json
import logging
import os
import re
import sys
import time
import traceback
from io import BytesIO

import matplotlib.pyplot as plt
import streamlit as st

import getdist
from getdist import IniFile, plots
from getdist.chain_grid import ChainDirGrid, get_chain_root_files

# The app will look for a default_chains directory in the following locations:
# 1. The directory containing this file
# 2. The parent directory of this file
# 3. The grandparent directory of this file (repository root)
# If found, it will be automatically selected when the app starts
#
# The app also accepts a command line argument to specify a default directory:
#
# When running with streamlit directly:
#   streamlit run streamlit_app.py -- --dir=/path/to/chains
#   streamlit run streamlit_app.py -- --directory=/path/to/chains
#
# When running the module with python -m:
#   python -m streamlit run getdist/gui/streamlit_app.py --dir=/path/to/chains
#   python -m streamlit run getdist/gui/streamlit_app.py --directory=/path/to/chains
#
# Both formats with and without the equals sign are supported:
#   --dir=/path/to/chains
#   --dir /path/to/chains

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "getdist_streamlit.log")),
    ],
)
logger = logging.getLogger(__name__)


def track_session_reload():
    """Track why the session is reloading"""
    if "reload_count" not in st.session_state:
        st.session_state.reload_count = 0

    st.session_state.reload_count += 1

    logger.info(
        f"""
Session reload #{st.session_state.reload_count}
Time: {time.strftime("%Y-%m-%d %H:%M:%S")}
Stack trace:
{traceback.format_stack()}
    """.strip()
    )


# Add near start of main()
track_session_reload()


def parse_command_line_args():
    dir_path = None
    # First try to find arguments after -- separator (for streamlit run -- --dir=path syntax)
    try:
        separator_index = sys.argv.index("--")
        # Get arguments after --
        args = sys.argv[separator_index + 1 :]
        # Look for --dir or --directory argument
        for i, arg in enumerate(args):
            if arg.startswith("--dir=") or arg.startswith("--directory="):
                # Extract the directory path
                parts = arg.split("=", 1)
                if len(parts) == 2 and parts[1]:
                    dir_path = os.path.abspath(parts[1])
                    break
            elif (arg == "--dir" or arg == "--directory") and i + 1 < len(args):
                # Directory is the next argument
                dir_path = os.path.abspath(args[i + 1])
                break
    except (ValueError, IndexError):
        # -- not found or other parsing error
        args = sys.argv
        for i, arg in enumerate(args):
            if arg.startswith("--dir=") or arg.startswith("--directory="):
                # Extract the directory path
                parts = arg.split("=", 1)
                if len(parts) == 2 and parts[1]:
                    dir_path = os.path.abspath(parts[1])
                    break
            elif (arg == "--dir" or arg == "--directory") and i + 1 < len(args):
                # Directory is the next argument
                dir_path = os.path.abspath(args[i + 1])
                break
    if dir_path:
        logger.info("Found directory argument: %s", dir_path)
    return dir_path


# Set page configuration
st.set_page_config(page_title="GetDist GUI", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

default_plot_settings = {
    # General settings
    "color_by": "None",
    # 1D plot settings
    "normalized": False,
    # 2D plot settings
    "filled": True,
    "shaded": False,
    "axis_legend": False,
    # Z-axis settings
    "use_z_axis": False,
    "z_param": None,
    "shadows": False,
    # Triangle plot settings
    "show_1d": False,
}

# --- Session State Initialization ---

# Core Data & Paths
st.session_state.setdefault("chain_dir", None)
st.session_state.setdefault("current_dir", None)
st.session_state.setdefault("batch", None)
st.session_state.setdefault("plotter", None)
st.session_state.setdefault("selected_roots", [])
st.session_state.setdefault("param_names", None)
st.session_state.setdefault("root_infos", {})
st.session_state.setdefault("active_chain", None)

# File Browser State
st.session_state.setdefault("current_browser_path", os.path.expanduser("~"))
st.session_state.setdefault("recent_directories", [])
st.session_state.setdefault("show_file_browser", False)
st.session_state.setdefault("selected_chain_to_add", None)
st.session_state.setdefault("previous_chain_selection", None)
st.session_state.setdefault("display_dir_path", None)

# Initialize selected_directory with None first
st.session_state.setdefault("selected_directory", None)

# Check for command line arguments or default_chains folder if app is starting fresh
if "app_initialized" not in st.session_state:
    # First check for command line arguments
    cmd_line_dir = parse_command_line_args()
    if cmd_line_dir and os.path.exists(cmd_line_dir) and os.path.isdir(cmd_line_dir):
        logger.info(f"Using directory from command line argument: {cmd_line_dir}")
        st.session_state.selected_directory = os.path.abspath(cmd_line_dir)
    else:
        # If no command line argument, look for default_chains in current directory, parent directory, and grandparent directory
        file_dir = os.path.dirname(__file__)
        logger.info(f"File directory: {file_dir}")
        possible_locations = [
            "",
            file_dir,  # Current directory
            os.path.abspath(os.path.join(file_dir, "..")),  # Parent directory
            os.path.abspath(os.path.join(file_dir, "..", "..")),  # Grandparent directory
        ]

        # Try to find default_chains in any of the possible locations
        for location in possible_locations:
            test_path = os.path.join(location, "default_chains")
            if os.path.exists(test_path) and os.path.isdir(test_path):
                default_chains_dir = test_path
                logger.info(f"Found default_chains directory at: {default_chains_dir}")
                st.session_state.selected_directory = os.path.abspath(default_chains_dir)
                break

    # Mark that the app has been initialized
    st.session_state.app_initialized = True

# Plotting State & Parameters
st.session_state.setdefault("plot_type", "1D Density")
st.session_state.setdefault("x_params", [])
st.session_state.setdefault("y_params", [])
st.session_state.setdefault("plot_settings", default_plot_settings)
st.session_state.setdefault("current_plot", None)
st.session_state.setdefault("force_replot", False)
st.session_state.setdefault("plot_module", "getdist.plots")
st.session_state.setdefault("script_plot_module", "getdist.plots")

# Scripting State
st.session_state.setdefault("current_script", "")

# Dialog/UI Visibility States
st.session_state.setdefault("show_marge_stats", False)
st.session_state.setdefault("show_like_stats", False)
st.session_state.setdefault("show_converge_stats", False)
st.session_state.setdefault("show_pca", False)
st.session_state.setdefault("show_param_table", False)
st.session_state.setdefault("show_analysis_settings", False)
st.session_state.setdefault("show_plot_options", False)
st.session_state.setdefault("show_config_settings", False)
st.session_state.setdefault("show_about", False)
st.session_state.setdefault("show_debug_log", False)

# GetDist Settings Initialization
if "default_settings" not in st.session_state:
    st.session_state.default_settings = IniFile(getdist.default_getdist_settings)
    logger.info("Initialized default GetDist settings.")

if "base_settings" not in st.session_state:
    st.session_state.base_settings = st.session_state.default_settings
    logger.info("Initialized base settings from defaults.")

if "current_settings" not in st.session_state:
    # Deep copy to ensure modifications don't affect the base settings
    st.session_state.current_settings = copy.deepcopy(st.session_state.base_settings)
    logger.info("Initialized current settings from base settings.")


# Functions for persisting recent directories between sessions
def get_config_dir():
    """Get the directory for storing configuration files"""
    # Use the same directory as getdist.gui would use
    # Windows: %APPDATA%\getdist\streamlit
    # Unix/Linux/Mac: ~/.config/getdist/streamlit
    base_dir = (
        os.environ.get("APPDATA", os.path.expanduser("~"))
        if os.name == "nt"
        else os.path.join(os.path.expanduser("~"), ".config")
    )
    config_dir = os.path.join(base_dir, "getdist", "streamlit")

    # Create the directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    return config_dir


def get_recent_dirs_file():
    """Get the path to the file storing recent directories"""
    return os.path.join(get_config_dir(), "recent_directories.json")


def save_recent_directories():
    """Save the list of recent directories to a file"""
    try:
        recent_dirs = st.session_state.recent_directories
        if recent_dirs:
            # Limit to 10 most recent directories
            if len(recent_dirs) > 10:
                recent_dirs = recent_dirs[:10]

            # Save to file
            with open(get_recent_dirs_file(), "w") as f:
                json.dump(recent_dirs, f)
            logger.info(f"Saved {len(recent_dirs)} recent directories to {get_recent_dirs_file()}")
    except Exception as e:
        logger.error(f"Error saving recent directories: {str(e)}")


def load_recent_directories():
    """Load the list of recent directories from a file"""
    try:
        file_path = get_recent_dirs_file()
        if os.path.exists(file_path):
            with open(file_path) as f:
                recent_dirs = json.load(f)

            # Validate the loaded data
            if isinstance(recent_dirs, list):
                # Filter out directories that no longer exist
                recent_dirs = [d for d in recent_dirs if os.path.exists(d)]
                st.session_state.recent_directories = recent_dirs
                logger.info(f"Loaded {len(recent_dirs)} recent directories from {file_path}")
    except Exception as e:
        logger.error(f"Error loading recent directories: {str(e)}")


# Load recent directories when the app starts
load_recent_directories()


def get_plotter(chain_dir=None):
    """Get a subplot plotter instance

    Args:
        chain_dir: Directory containing chains
    """

    chain_dirs = []
    chain_dir = chain_dir or st.session_state.chain_dir
    if chain_dir:
        chain_dirs.append(chain_dir)
    for root in st.session_state.root_infos:
        info = st.session_state.root_infos[root]
        if info.batch:
            if info.batch not in chain_dirs:
                chain_dirs.append(info.batch)

    st.session_state.plotter = plots.get_subplot_plotter(
        chain_dir=chain_dirs, analysis_settings=st.session_state.current_settings
    )
    return st.session_state.plotter


def open_directory(dir_path):
    """Open a directory containing chains"""
    if not os.path.exists(dir_path):
        st.error(f"Directory not found: {dir_path}")
        return False

    # Update the display_dir_path variable instead of dir_path_input
    # This will be used to update the input field on the next rerun
    st.session_state.display_dir_path = dir_path

    try:
        # Try to read as a directory with chain files
        root_list = get_chain_root_files(dir_path)
        if root_list:
            st.session_state.chain_dir = dir_path
            st.session_state.current_dir = dir_path
            st.session_state.batch = None
            get_plotter(chain_dir=dir_path)
            return True

        # Try to read as a directory with subdirectories containing chains
        try:
            batch = ChainDirGrid(dir_path)
            if batch.base_dir_names:
                st.session_state.chain_dir = dir_path
                st.session_state.batch = batch
                st.session_state.current_dir = dir_path
                get_plotter(chain_dir=batch)
                return True
        except Exception as e:
            logging.warning(f"Not a ChainDirGrid: {str(e)}")

        st.error(f"No chains found in {dir_path}")
        return False

    except Exception as e:
        st.error(f"Error opening directory: {str(e)}")
        return False


def update_parameters():
    """Update parameter names by merging from all selected roots

    This function mirrors the _updateParameters function in the Qt implementation.
    It creates a unified parameter list by:
    1. Starting with the first root's parameters
    2. Adding renames from all other roots
    3. Handling potential renames to ensure consistent parameter naming across chains
    """
    roots = st.session_state.selected_roots
    if not roots:
        logging.warning("No roots selected for parameter update")
        return

    logging.info(f"Updating parameters from roots: {roots}")

    # Save current parameter selections before updating
    old_x_params = st.session_state.x_params.copy()
    old_y_params = st.session_state.y_params.copy()

    try:
        # Get plotter to access samples
        plotter = get_plotter()
        if not plotter:
            logging.error("Failed to get plotter for parameter update")
            return

        # Get samples for the first root
        samples = plotter.sample_analyser.samples_for_root(roots[0])
        if not samples or not hasattr(samples, "paramNames"):
            logging.error(f"No valid samples or paramNames found for {roots[0]}")
            return

        # Create a copy of the first root's paramNames (we don't want to change the original)
        # Use the filteredCopy method if available, otherwise create a new reference
        if hasattr(samples.paramNames, "filteredCopy"):
            # This matches the Qt implementation exactly
            param_names = samples.paramNames.filteredCopy(samples.paramNames)
            logging.info(f"Created filtered copy of paramNames from {roots[0]}")
        else:
            # Fallback if filteredCopy is not available
            param_names = samples.paramNames
            logging.info(f"Using reference to paramNames from {roots[0]}")

        # Add renames from all other roots
        for root in roots[1:]:
            try:
                other_samples = plotter.sample_analyser.samples_for_root(root)
                if other_samples and hasattr(other_samples, "getRenames"):
                    # Update renames from this root
                    param_names.updateRenames(other_samples.getRenames())
                    logging.info(f"Updated renames from {root}")
                else:
                    logging.warning(f"Could not get renames from {root}")
            except Exception as e:
                logging.error(f"Error updating renames from {root}: {str(e)}")

        # Store the updated parameter names in session state
        st.session_state.param_names = param_names
        logging.info("Parameter names updated successfully")

        # Update parameter selections to maintain consistency
        # This is similar to the Qt implementation's _updateListParametersSelection
        if hasattr(param_names, "parWithName"):
            # Update X parameters
            new_x_params = []
            for param in old_x_params:
                # Try to find the parameter in the new parameter list
                param_info = param_names.parWithName(param, error=False)
                if param_info:
                    new_x_params.append(param_info.name)
            st.session_state.x_params = new_x_params

            # Update Y parameters
            new_y_params = []
            for param in old_y_params:
                # Try to find the parameter in the new parameter list
                param_info = param_names.parWithName(param, error=False)
                if param_info:
                    new_y_params.append(param_info.name)
            st.session_state.y_params = new_y_params

            logging.info(f"Updated parameter selections: X={new_x_params}, Y={new_y_params}")
    except Exception as e:
        logging.error(f"Error updating parameters: {str(e)}")


def add_root(root_name):
    """Add a root to the selected roots"""
    if not root_name:
        logging.warning("Attempted to add empty root name")
        return

    if root_name in st.session_state.selected_roots:
        logging.info(f"Root {root_name} already in selected roots")
        return

    logging.info(f"Adding root: {root_name}")
    st.session_state.selected_roots.append(root_name)

    # Set as active chain
    st.session_state.active_chain = root_name

    # Create RootInfo and add to plotter
    plotter = get_plotter()
    if not plotter:
        logging.error("Failed to get plotter")
        return

    try:
        # Get the path for the root
        if st.session_state.batch:
            logging.info(f"Getting path from batch for {root_name}")
            try:
                if hasattr(st.session_state.batch, "resolve_root"):
                    path = st.session_state.batch.resolve_root(root_name).chainPath
                    logging.info(f"Using resolve_root, path: {path}")
                else:
                    path = st.session_state.batch.resolveRoot(root_name).chainPath
                    logging.info(f"Using resolveRoot, path: {path}")
            except Exception as e:
                logging.error(f"Error resolving root path: {str(e)}")
                path = st.session_state.chain_dir
        else:
            path = st.session_state.chain_dir
            logging.info(f"Using chain_dir as path: {path}")

        if root_name[-1] in (os.sep, "/"):
            path = os.sep.join(path.replace("/", os.sep).split(os.sep)[:-1])

        # Create RootInfo
        logging.info(f"Creating RootInfo for {root_name} with path {path}")
        info = plots.RootInfo(root_name, path, st.session_state.batch)

        # Add to plotter
        logging.info(f"Adding root {root_name} to plotter")
        plotter.sample_analyser.add_root(info)

        # Store in session state
        st.session_state.root_infos[root_name] = info

        # Update parameters by merging from all selected roots
        # This replaces the previous parameter update logic
        update_parameters()
    except Exception as e:
        logging.error(f"Error adding root {root_name}: {str(e)}")
        # Remove from selected roots if there was an error
        if root_name in st.session_state.selected_roots:
            st.session_state.selected_roots.remove(root_name)
        st.error(f"Error adding chain {root_name}: {str(e)}")


def show_marge_stats(rootname=None):
    """Show marginalized statistics for the selected chain"""
    if not st.session_state.selected_roots:
        st.warning("Please select a chain first")
        return

    # If no specific rootname is provided, use the currently active chain
    if rootname is None:
        if "active_chain" in st.session_state and st.session_state.active_chain in st.session_state.selected_roots:
            rootname = st.session_state.active_chain
        else:
            rootname = st.session_state.selected_roots[0]

    try:
        plotter = get_plotter()
        if not plotter:
            st.error("Failed to get plotter")
            return

        samples = plotter.sample_analyser.samples_for_root(rootname)
        if not samples:
            st.error(f"Failed to get samples for {rootname}")
            return

        stats = samples.getMargeStats()
        if not stats:
            st.error(f"Failed to get marginalized statistics for {rootname}")
            return

        return rootname, stats
    except Exception as e:
        st.error(f"Error getting marginalized statistics: {str(e)}")
        return None


def show_like_stats(rootname=None):
    """Show likelihood statistics for the selected chain"""
    if not st.session_state.selected_roots:
        st.warning("Please select a chain first")
        return

    # If no specific rootname is provided, use the currently active chain
    if rootname is None:
        if "active_chain" in st.session_state and st.session_state.active_chain in st.session_state.selected_roots:
            rootname = st.session_state.active_chain
        else:
            rootname = st.session_state.selected_roots[0]

    try:
        plotter = get_plotter()
        if not plotter:
            st.error("Failed to get plotter")
            return

        samples = plotter.sample_analyser.samples_for_root(rootname)
        if not samples:
            st.error(f"Failed to get samples for {rootname}")
            return

        stats = samples.getLikeStats()
        if not stats:
            st.error(f"Failed to get likelihood statistics for {rootname}")
            return

        return rootname, stats
    except Exception as e:
        st.error(f"Error getting likelihood statistics: {str(e)}")
        return None


def show_converge_stats(rootname=None):
    """Show convergence statistics for the selected chain"""
    if not st.session_state.selected_roots:
        st.warning("Please select a chain first")
        return

    # If no specific rootname is provided, use the currently active chain
    if rootname is None:
        if "active_chain" in st.session_state and st.session_state.active_chain in st.session_state.selected_roots:
            rootname = st.session_state.active_chain
        else:
            rootname = st.session_state.selected_roots[0]

    try:
        plotter = get_plotter()
        if not plotter:
            st.error("Failed to get plotter")
            return

        samples = plotter.sample_analyser.samples_for_root(rootname)
        if not samples:
            st.error(f"Failed to get samples for {rootname}")
            return

        # Get the converge_test_limit from settings
        converge_test_limit = float(st.session_state.analysis_settings.get("converge_test_limit", "0.2"))

        stats = samples.getConvergeTests(converge_test_limit)
        if stats is None:
            st.error(f"Failed to get convergence statistics for {rootname}")
            return

        summary = samples.getNumSampleSummaryText()
        if hasattr(samples, "GelmanRubin"):
            summary += "\nvar(mean)/mean(var), remaining chains, worst e-value: R-1 = %13.5F" % samples.GelmanRubin

        return rootname, stats, summary
    except Exception as e:
        st.error(f"Error getting convergence statistics: {str(e)}")
        return None


def show_pca(rootname=None):
    """Show PCA analysis for selected parameters"""
    if not st.session_state.selected_roots or not st.session_state.x_params:
        st.warning("Please select a chain and parameters first")
        return

    # If no specific rootname is provided, use the currently active chain
    if rootname is None:
        if "active_chain" in st.session_state and st.session_state.active_chain in st.session_state.selected_roots:
            rootname = st.session_state.active_chain
        else:
            rootname = st.session_state.selected_roots[0]

    try:
        plotter = get_plotter()
        if not plotter:
            st.error("Failed to get plotter")
            return

        samples = plotter.sample_analyser.samples_for_root(rootname)
        if not samples:
            st.error(f"Failed to get samples for {rootname}")
            return

        # Get PCA for selected parameters
        pca_text = samples.PCA(st.session_state.x_params)
        if not pca_text:
            st.error(f"Failed to perform PCA for {rootname}")
            return

        # Extract eigenvalues
        eigenvalues = []
        eigenvalue_pattern = r"PC\s*(\d+):\s*([\d\.]+)"
        eigenvalue_matches = re.findall(eigenvalue_pattern, pca_text)
        for _, val in eigenvalue_matches:
            eigenvalues.append(float(val))

        # Extract eigenvectors
        eigenvectors = []
        # Find the section with e-vectors
        evector_section = re.search(r"e-vectors\n([\s\S]*?)\n\n", pca_text)
        if evector_section:
            evector_text = evector_section.group(1)
            evector_lines = evector_text.strip().split("\n")
            for line in evector_lines:
                # Extract the vector values (skipping the parameter index at the start)
                values = re.findall(r"([\-\d\.]+)", line)[1:]
                eigenvectors.append([float(v) for v in values])

        return rootname, (eigenvalues, eigenvectors, pca_text)
    except Exception as e:
        st.error(f"Error performing PCA: {str(e)}")
        return None


def reload_files():
    """Reload chain files from the current directory"""
    if not st.session_state.chain_dir:
        st.warning("No directory selected")
        return False

    # Clear current selections
    st.session_state.selected_roots = []
    st.session_state.active_chain = None
    st.session_state.param_names = None
    st.session_state.x_params = []
    st.session_state.y_params = []
    st.session_state.current_plot = None
    st.session_state.current_script = ""
    st.session_state.root_infos = {}

    # Reopen the directory
    return open_directory(st.session_state.chain_dir)


def apply_analysis_settings(settings):
    """Apply analysis settings to the plotter"""
    if not st.session_state.plotter:
        st.warning("No plotter available")
        return

    # Update current settings
    st.session_state.current_settings.params.update(settings)

    # Reset the sample analyzer with the new settings
    try:
        # Reset the plotter with new settings
        st.session_state.plotter.sample_analyser.reset(
            st.session_state.current_settings, chain_settings_have_priority=False
        )

        # Update the current plot if one exists
        if st.session_state.current_plot:
            # Regenerate the plot with the new settings
            # This matches the original mainwindow.py settingsChanged method
            # which calls plotData() to regenerate the plot

            # Store the current plot type and parameters to regenerate the plot
            if hasattr(st.session_state, "plot_type"):
                st.session_state.force_replot = True

                # Don't clear the figure here - we'll regenerate it completely
                # in the main function when force_replot is True

                # Apply the settings and trigger a rerun
                st.rerun()

        st.success("Analysis settings applied")
    except Exception as e:
        st.error(f"Error applying settings: {str(e)}")


def apply_plot_module(module_name):
    """Apply plot style module"""
    if not module_name:
        return

    try:
        # Update session state
        st.session_state.plot_module = module_name

        # Import the module dynamically
        import importlib

        try:
            style_module = importlib.import_module(module_name)
            # Apply the style if the module has a get_subplot_plotter function
            if hasattr(style_module, "get_subplot_plotter"):
                st.session_state.plotter = style_module.get_subplot_plotter(chain_dir=st.session_state.chain_dir)
                # Store the default plot settings
                st.session_state.default_plot_settings = copy.copy(st.session_state.plotter.settings)
                st.success(f"Applied plot style module: {module_name}")
            else:
                st.warning(f"Module {module_name} does not have get_subplot_plotter function")
        except ImportError:
            st.error(f"Could not import module: {module_name}")
    except Exception as e:
        st.error(f"Error applying plot style module: {str(e)}")


def reset_analysis_settings():
    """Reset analysis settings to defaults"""
    st.session_state.current_settings = copy.deepcopy(st.session_state.base_settings)

    # Reset the sample analyzer with the default settings
    if st.session_state.plotter:
        try:
            st.session_state.plotter.sample_analyser.reset(
                st.session_state.current_settings, chain_settings_have_priority=False
            )
            st.success("Analysis settings reset to defaults")
        except Exception as e:
            st.error(f"Error resetting settings: {str(e)}")
    else:
        st.success("Analysis settings reset to defaults")


def reset_plot_options():
    """Reset plot options to defaults"""
    st.session_state.plot_settings = default_plot_settings.copy()
    st.success("Plot options reset to defaults")


def changed_settings():
    base = st.session_state.base_settings.params
    current = st.session_state.current_settings.params
    return {key: value for key, value in current.items() if base[key] != value}


def set_size_for_n(plotter, cols, rows, width_inch, height_inch):
    """Set figure width and subplot ratio based on grid dimensions"""
    # Set figure width (use existing or new width, whichever is smaller)
    plotter.settings.fig_width_inch = min(plotter.settings.fig_width_inch or width_inch, width_inch)

    # Calculate aspect ratio factor
    aspect_factor = height_inch * cols / rows

    # Set or adjust subplot size ratio
    if plotter.settings.subplot_size_ratio:
        plotter.settings.fig_width_inch = min(
            plotter.settings.fig_width_inch, aspect_factor / plotter.settings.subplot_size_ratio
        )
    else:
        plotter.settings.subplot_size_ratio = min(1.5, aspect_factor / plotter.settings.fig_width_inch)

    logging.info(
        f"Set plot size: {cols}x{rows} grid, width={plotter.settings.fig_width_inch:.2f}in, ratio={plotter.settings.subplot_size_ratio:.2f}"
    )


def export_plot(format_type):
    """Export the current plot in the specified format without using temporary files when possible

    Args:
        format_type: The format to export ("PNG" or "PDF")

    Returns:
        tuple: (file_data, mime_type) where file_data is the binary data and mime_type is the MIME type
    """
    logging.info(f"Exporting plot in {format_type} format")

    # Check if we have a plotter with a figure
    if (
        not hasattr(st.session_state, "plotter")
        or not st.session_state.plotter
        or not hasattr(st.session_state.plotter, "fig")
    ):
        logging.warning("No plotter or figure available for export")
        # Fallback to PNG from memory if plotter not available
        if format_type == "PNG" and st.session_state.current_plot:
            return st.session_state.current_plot, "image/png"
        return None, None

    try:
        plotter = st.session_state.plotter

        # Set the appropriate mime type
        if format_type == "PNG":
            mime_type = "image/png"
            # For PNG, we can use BytesIO directly without temporary files
            buf = BytesIO()
            plotter.fig.savefig(
                buf,
                format="png",
                edgecolor="w",
                facecolor="w",
                dpi=300,  # Higher DPI for better quality
                bbox_inches="tight",
            )
            buf.seek(0)
            return buf.getvalue(), mime_type

        elif format_type == "PDF":
            mime_type = "application/pdf"
            # For PDF, use the plotter's export method to ensure consistency with original GetDist GUI
            # This requires a temporary file since the export() method writes to a file
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_name = tmp.name

            try:
                # Use the plotter's export function with the temporary file
                plotter.export(tmp_name)

                # Read the exported file
                with open(tmp_name, "rb") as f:
                    file_data = f.read()

                # Clean up the temporary file
                os.unlink(tmp_name)

                return file_data, mime_type
            except Exception as e:
                logging.exception(f"Error using plotter.export: {str(e)}")
                # Clean up the temporary file if it exists
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

                # Fallback to direct PDF generation
                logging.info("Falling back to direct PDF generation")
                buf = BytesIO()
                plotter.fig.savefig(buf, format="pdf", edgecolor="w", facecolor="w", bbox_inches="tight")
                buf.seek(0)
                return buf.getvalue(), mime_type

        else:
            logging.error(f"Unsupported export format: {format_type}")
            return None, None

    except Exception as e:
        logging.exception(f"Error exporting plot: {str(e)}")
        return None, None


def generate_plot():
    """Generate a plot based on current selections"""
    logging.info("Starting plot generation")

    if not st.session_state.selected_roots:
        logging.warning("No roots selected for plotting")
        return None, None

    if not st.session_state.x_params:
        logging.warning("No X parameters selected for plotting")
        return None, None

    logging.info(
        f"Generating plot with roots: {st.session_state.selected_roots} and params: {st.session_state.x_params}"
    )

    # Properly close all existing figures to prevent overlapping plots
    plt.close("all")
    plt.clf()

    # Get a fresh plotter instance
    plotter = get_plotter()
    if not plotter:
        logging.error("Failed to get plotter for plot generation")
        return None, None

    # Set default figure size based on plot type - similar to Qt implementation
    # In Qt, this is based on widget dimensions, but in Streamlit we need to use reasonable defaults
    # Default width and height in inches (similar to Qt's logical DPI scaling)
    width_inch = 8  # Default width in inches
    height_inch = 6  # Default height in inches

    # Generate script
    script_lines = ["from getdist import plots", ""]

    if st.session_state.script_plot_module != "getdist.plots":
        script_lines.append(
            "from %s import style_name\nplots.set_active_style(style_name)" % st.session_state.script_plot_module
        )
        script_lines += []

    if override_setting := changed_settings():
        script_lines.append(("analysis_settings = %s\n" % override_setting).replace(", ", ",\n" + " " * 21))

    # Create plotter - use raw string for path as in original mainwindow.py
    # Follow the exact same plot_func logic as in mainwindow.py
    if len(st.session_state.x_params) > 1 or len(st.session_state.y_params) > 1:
        plot_func = "get_subplot_plotter("
        # Add subplot_size parameter in specific cases as in the original
        if (
            not plotter.settings.fig_width_inch
            and len(st.session_state.y_params)
            and not (len(st.session_state.x_params) > 1 and len(st.session_state.y_params) > 1)
            and st.session_state.plot_type != "Triangle"
        ):
            plot_func += "subplot_size=3.5, "
    else:
        plot_func = "get_single_plotter("

    # Format chain_dirs with r'' prefix for Windows paths, matching mainwindow.py
    chain_dirs = st.session_state.chain_dir
    if isinstance(chain_dirs, str):
        chain_dirs = "r'%s'" % chain_dirs.rstrip("\\").rstrip("/")

    if override_setting:
        script_lines += [f"g=plots.{plot_func}chain_dir={chain_dirs},analysis_settings=analysis_settings)"]
    else:
        script_lines += [f"g=plots.{plot_func}chain_dir={chain_dirs})"]

    script_lines.append("")

    # Add roots - use the exact same format as in original mainwindow.py
    roots_str = ", ".join([f"'{root}'" for root in st.session_state.selected_roots])
    script_lines.append(f"roots = [{roots_str}]")
    script_lines.append("")

    # Generate plot based on plot type
    fig = None
    logging.info(f"Plot type: {st.session_state.plot_type}")

    if st.session_state.plot_type == "1D Density":
        # Check if we have X parameters
        if not st.session_state.x_params:
            logging.warning("No X parameters selected for 1D plot")
            st.error("Please select at least one X parameter for the 1D plot.")
            return None, "# Error: No X parameters selected for 1D plot"
        # 1D plot
        params_str = ", ".join([f"'{param}'" for param in st.session_state.x_params])
        normalized = st.session_state.plot_settings.get("normalized", False)

        # Format exactly like original mainwindow.py
        script_lines.append(f"params=[{params_str}]")
        script_lines.append("g.plots_1d(roots, params=params)")

        try:
            logging.info("Using existing plotter for 1D plot")
            logging.info(
                f"Calling plots_1d with roots={st.session_state.selected_roots}, params={st.session_state.x_params}, normalized={normalized}"
            )

            # Set appropriate figure size for 1D plots - similar to Qt implementation
            if st.session_state.plot_type == "1D Density":
                # Calculate appropriate figure dimensions based on number of parameters
                cols, rows = plotter.default_col_row(len(st.session_state.x_params))

                # Use the set_size_for_n function to set the figure size
                set_size_for_n(plotter, cols, rows, width_inch, height_inch)

            # Call plots_1d on the existing plotter
            plotter.plots_1d(
                st.session_state.selected_roots, st.session_state.x_params, normalized=normalized, colors=None
            )

            # plots_1d already handles tight_layout internally

            # Get the figure directly from the plotter
            logging.info("Getting figure from plotter.fig")
            fig = plotter.fig

            # No title in the original app

            logging.info("Successfully generated 1D plot")
        except Exception as e:
            logging.exception("Error generating 1D plot")
            st.error(f"Error generating 1D plot: {str(e)}")

    elif st.session_state.plot_type == "2D Contour":
        # Check if we have both X and Y parameters
        if len(st.session_state.x_params) < 1:
            logging.warning("No X parameter selected for 2D plot")
            st.error("Please select at least one X parameter for the 2D plot.")
            return None, "# Error: No X parameter selected for 2D plot"

        if len(st.session_state.y_params) < 1:
            logging.warning("No Y parameter selected for 2D plot")
            st.error("Please select at least one Y parameter for the 2D plot.")
            return None, "# Error: No Y parameter selected for 2D plot"
        # Check if x and y parameters are the same
        if st.session_state.x_params[0] == st.session_state.y_params[0]:
            logging.warning(f"Cannot create 2D plot with same parameter for x and y: {st.session_state.x_params[0]}")
            st.error(
                f"Cannot create 2D plot with same parameter for x and y: {st.session_state.x_params[0]}. Please select different parameters."
            )
            return None, "# Error: Cannot create 2D plot with same parameter for x and y"
        # 2D plot
        filled = st.session_state.plot_settings.get("filled", True)

        # Get color_by parameter if set
        color_by = st.session_state.plot_settings.get("color_by", None)
        if color_by == "None":
            color_by = None

        # Check if Z-axis is enabled
        use_z_axis = st.session_state.plot_settings.get("use_z_axis", False)
        z_param = st.session_state.plot_settings.get("z_param", None)
        shadows = st.session_state.plot_settings.get("shadows", False)

        if use_z_axis and z_param and len(st.session_state.x_params) == 1 and len(st.session_state.y_params) == 1:
            # Generate 4D plot script (x-y-z with optional color)
            params = [st.session_state.x_params[0], st.session_state.y_params[0], z_param]

            # Add color parameter if set
            if color_by and color_by != "None":
                params.append(color_by)

            # Format exactly like original mainwindow.py
            script_lines.append(f"params = {str(params)}")

            # Add colors for multiple chains - match original format exactly
            if len(st.session_state.selected_roots) > 1:
                script_lines.append("colors = [c[-1] for c in g.settings.line_styles[:len(roots) - 1]]")
                shadow_str = ", shadow_color=True" if shadows else ""
                script_lines.append(
                    f"g.plot_4d(roots, params, color_bar=True{'' if len(st.session_state.selected_roots) == 1 else ', compare_colors=colors'}{shadow_str})"
                )
            else:
                shadow_str = ", shadow_color=True" if shadows else ""
                script_lines.append(f"g.plot_4d(roots, params, color_bar=True{shadow_str})")
        elif color_by and color_by != "None":
            # Generate 3D scatter plot script - match original format exactly
            triplet = f"['{st.session_state.x_params[0]}', '{st.session_state.y_params[0]}', '{color_by}']"
            script_lines.append("g.settings.scatter_size = 6")
            script_lines.append("g.make_figure()")
            script_lines.append(f"g.plot_3d(roots, {triplet})")
        else:
            # Generate regular 2D contour plot script - match original format exactly
            shaded = st.session_state.plot_settings.get("shaded", False)

            # Handle different combinations of X and Y parameters - follow mainwindow.py logic exactly
            single = False

            if len(st.session_state.x_params) == 1 and len(st.session_state.y_params) == 1:
                # Single pair
                pairs = [[st.session_state.x_params[0], st.session_state.y_params[0]]]
                single = st.session_state.plot_settings.get("axis_legend", False)

                # For single pairs with axis_legend, add legend after plot
                if single:
                    script_lines.append(
                        f"g.plot_2d(roots, '{pairs[0][0]}', '{pairs[0][1]}', filled={filled}, shaded={shaded})"
                    )
                    script_lines.append("labels = g._default_legend_labels(None, roots)")
                    script_lines.append("g.add_legend(labels)")
                else:
                    script_lines.append(
                        f"g.plot_2d(roots, '{pairs[0][0]}', '{pairs[0][1]}', filled={filled}, shaded={shaded})"
                    )

            elif len(st.session_state.x_params) == 1 and len(st.session_state.y_params) > 1:
                # One X, multiple Y
                item_x = st.session_state.x_params[0]
                pairs = [[item_x, y] for y in st.session_state.y_params]
                script_lines.append(f"pairs = {pairs}")
                script_lines.append(f"g.plots_2d(roots, param_pairs=pairs, filled={filled}, shaded={shaded})")

            elif len(st.session_state.x_params) > 1 and len(st.session_state.y_params) == 1:
                # Multiple X, one Y
                item_y = st.session_state.y_params[0]
                pairs = [[x, item_y] for x in st.session_state.x_params]
                script_lines.append(f"pairs = {pairs}")
                script_lines.append(f"g.plots_2d(roots, param_pairs=pairs, filled={filled}, shaded={shaded})")

            elif len(st.session_state.x_params) > 1 and len(st.session_state.y_params) > 1:
                # Rectangle plot - multiple X and Y parameters
                script_lines.append(f"xparams = {st.session_state.x_params}")
                script_lines.append(f"yparams = {st.session_state.y_params}")
                script_lines.append(f"g.rectangle_plot(xparams, yparams, roots=roots, filled={filled})")

        try:
            # Get settings
            filled = st.session_state.plot_settings.get("filled", True)

            logging.info(
                f"Calling plot_2d with roots={st.session_state.selected_roots}, x={st.session_state.x_params[0]}, y={st.session_state.y_params[0]}, filled={filled}"
            )

            # Get color_by parameter if set
            color_by = st.session_state.plot_settings.get("color_by", None)
            if color_by == "None":
                color_by = None

            # Use the existing plotter that already has the samples loaded
            logging.info("Using existing plotter for 2D plot")

            # Set appropriate figure size for 2D plots - similar to Qt implementation
            if st.session_state.plot_type == "2D Contour":
                # Get settings
                filled = st.session_state.plot_settings.get("filled", True)
                shaded = st.session_state.plot_settings.get("shaded", False)

                # Handle different combinations of X and Y parameters
                if len(st.session_state.x_params) == 1 and len(st.session_state.y_params) == 1:
                    # Single pair - set size for 1x1 plot
                    set_size_for_n(plotter, 1, 1, width_inch, height_inch)

                elif len(st.session_state.x_params) == 1 and len(st.session_state.y_params) > 1:
                    # One X, multiple Y - calculate grid dimensions
                    pairs = [[st.session_state.x_params[0], y] for y in st.session_state.y_params]
                    cols, rows = plotter.default_col_row(len(pairs))

                    # Use the set_size_for_n function to set the figure size
                    set_size_for_n(plotter, cols, rows, width_inch, height_inch)

                elif len(st.session_state.x_params) > 1 and len(st.session_state.y_params) == 1:
                    # Multiple X, one Y - calculate grid dimensions
                    pairs = [[x, st.session_state.y_params[0]] for x in st.session_state.x_params]
                    cols, rows = plotter.default_col_row(len(pairs))

                    # Use the set_size_for_n function to set the figure size
                    set_size_for_n(plotter, cols, rows, width_inch, height_inch)

                elif len(st.session_state.x_params) > 1 and len(st.session_state.y_params) > 1:
                    # Rectangle plot - use x and y counts directly
                    cols = len(st.session_state.x_params)
                    rows = len(st.session_state.y_params)

                    # Use the set_size_for_n function to set the figure size
                    set_size_for_n(plotter, cols, rows, width_inch, height_inch)

            # Check if Z-axis is enabled
            use_z_axis = st.session_state.plot_settings.get("use_z_axis", False)
            z_param = st.session_state.plot_settings.get("z_param", None)
            shadows = st.session_state.plot_settings.get("shadows", False)
            shaded = st.session_state.plot_settings.get("shaded", False)

            # Call appropriate plot function based on settings
            if use_z_axis and z_param and len(st.session_state.x_params) == 1 and len(st.session_state.y_params) == 1:
                # Create a 4D plot with Z-axis
                logging.info(f"Creating 4D plot with Z-axis: {z_param}")
                params = [st.session_state.x_params[0], st.session_state.y_params[0], z_param]
                if color_by and color_by != "None":
                    params.append(color_by)

                # Set up colors for multiple chains
                if len(st.session_state.selected_roots) > 1:
                    colors = [c[-1] for c in plotter.settings.line_styles[: len(st.session_state.selected_roots) - 1]]
                    plotter.plot_4d(
                        st.session_state.selected_roots,
                        params,
                        color_bar=z_param,
                        compare_colors=colors,
                        shadow_color=shadows,
                    )
                else:
                    plotter.plot_4d(st.session_state.selected_roots, params, color_bar=z_param, shadow_color=shadows)
            elif color_by and color_by != "None":
                # Create a 3D scatter plot colored by the selected parameter
                logging.info(f"Creating 3D scatter plot colored by {color_by}")
                # Use plot_3d to match the original mainwindow.py behavior
                # Set scatter size to match original
                plotter.settings.scatter_size = 6
                # Make figure first as in the original
                plotter.make_figure()
                # Create the parameter triplet as in the original
                param_triplet = [st.session_state.x_params[0], st.session_state.y_params[0], color_by]
                # Call plot_3d with the triplet
                plotter.plot_3d(st.session_state.selected_roots, param_triplet)
            else:
                # Create a 2D contour plot
                logging.info("Creating 2D contour plot")
                # Handle different combinations of X and Y parameters - follow mainwindow.py logic exactly
                single = False

                if len(st.session_state.x_params) == 1 and len(st.session_state.y_params) == 1:
                    # Single pair
                    pairs = [[st.session_state.x_params[0], st.session_state.y_params[0]]]
                    single = st.session_state.plot_settings.get("axis_legend", False)
                    logging.info(f"Creating single pair 2D plot (axis_legend={single})")

                    # Always make a figure before calling plot_2d
                    plotter.make_figure(1)

                    if single:
                        # Make a single figure and add legend inside the plot
                        plotter.plot_2d(
                            st.session_state.selected_roots, pairs[0][0], pairs[0][1], filled=filled, shaded=shaded
                        )
                        labels = plotter._default_legend_labels(None, st.session_state.selected_roots)
                        plotter.add_legend(labels)
                    else:
                        # Just plot directly
                        plotter.plot_2d(
                            st.session_state.selected_roots, pairs[0][0], pairs[0][1], filled=filled, shaded=shaded
                        )

                elif len(st.session_state.x_params) == 1 and len(st.session_state.y_params) > 1:
                    # One X, multiple Y
                    logging.info(f"Creating 2D plot with 1 x param and {len(st.session_state.y_params)} y params")
                    item_x = st.session_state.x_params[0]
                    pairs = [[item_x, y] for y in st.session_state.y_params]
                    cols, rows = plotter.default_col_row(len(pairs))
                    plotter.make_figure(cols, rows)
                    plotter.plots_2d(st.session_state.selected_roots, param_pairs=pairs, filled=filled, shaded=shaded)

                elif len(st.session_state.x_params) > 1 and len(st.session_state.y_params) == 1:
                    # Multiple X, one Y
                    logging.info(f"Creating 2D plot with {len(st.session_state.x_params)} x params and 1 y param")
                    item_y = st.session_state.y_params[0]
                    pairs = [[x, item_y] for x in st.session_state.x_params]
                    cols, rows = plotter.default_col_row(len(pairs))
                    plotter.make_figure(cols, rows)
                    plotter.plots_2d(st.session_state.selected_roots, param_pairs=pairs, filled=filled, shaded=shaded)

                elif len(st.session_state.x_params) > 1 and len(st.session_state.y_params) > 1:
                    # Rectangle plot - multiple X and Y parameters
                    logging.info(
                        f"Creating rectangle plot with {len(st.session_state.x_params)} x params and {len(st.session_state.y_params)} y params"
                    )
                    plotter.make_figure(len(st.session_state.x_params), len(st.session_state.y_params))
                    plotter.rectangle_plot(
                        st.session_state.x_params,
                        st.session_state.y_params,
                        roots=st.session_state.selected_roots,
                        filled=filled,
                    )

            logging.info("Successfully generated 2D plot")
        except Exception as e:
            logging.exception("Error generating 2D plot")
            st.error(f"Error generating 2D plot: {str(e)}")

    elif st.session_state.plot_type == "Triangle" and st.session_state.x_params:
        # Check if we have enough parameters for a triangle plot
        if len(st.session_state.x_params) < 2:
            logging.warning(f"Need at least 2 parameters for triangle plot, got {len(st.session_state.x_params)}")
            st.error("Need at least 2 parameters for a triangle plot. Please select at least 2 parameters.")
            return None, "# Error: Need at least 2 parameters for a triangle plot"
        # Triangle plot
        filled = st.session_state.plot_settings.get("filled", True)
        show_1d = st.session_state.plot_settings.get("show_1d", True)
        params_str = ", ".join([f"'{param}'" for param in st.session_state.x_params])

        # Get color_by parameter if set
        color_by = st.session_state.plot_settings.get("color_by", None)
        if color_by == "None":
            color_by = None

        # Format exactly like original mainwindow.py
        script_lines.append(f"params = [{params_str}]")
        script_lines.append(
            f"g.triangle_plot(roots, params, filled={filled}"
            + (f", plot_3d_with_param='{color_by}'" if color_by and color_by != "None" else "")
            + (", shaded=True" if st.session_state.plot_settings.get("shaded", False) else "")
            + (f", no_1d_plots={not show_1d}" if not show_1d else "")
            + (f", title_limit={show_1d}" if show_1d else "")
            + ")"
        )

        try:
            # Get settings
            filled = st.session_state.plot_settings.get("filled", True)
            show_1d = st.session_state.plot_settings.get("show_1d", True)

            logging.info(
                f"Calling triangle_plot with roots={st.session_state.selected_roots}, params={st.session_state.x_params}, filled={filled}"
            )

            # Get color_by parameter if set
            color_by = st.session_state.plot_settings.get("color_by", None)
            if color_by == "None":
                color_by = None

            logging.info(
                f"Calling triangle_plot with roots={st.session_state.selected_roots}, params={st.session_state.x_params}, filled={filled}, plot_3d_with_param={color_by}"
            )

            # Call triangle_plot on the existing plotter - use plot_3d_with_param as in original
            shaded = st.session_state.plot_settings.get("shaded", False)

            # Call triangle_plot exactly as in the original mainwindow.py
            plotter.triangle_plot(
                st.session_state.selected_roots,
                st.session_state.x_params,
                plot_3d_with_param=color_by,
                filled=filled,
                shaded=shaded,
                no_1d_plots=not show_1d,
                title_limit=show_1d,
            )

            logging.info("Successfully generated triangle plot")
        except Exception as e:
            logging.exception("Error generating triangle plot")
            st.error(f"Error generating triangle plot: {str(e)}")

    script = "\n".join(script_lines)

    # Generate plot image in memory if figure is available
    if (fig := plotter.fig) is not None:
        logging.info("Saving figure to memory")
        try:
            # Create a BytesIO object to store the image
            buf = BytesIO()

            # Save the figure to memory - similar to how mainwindow.py does it
            fig.savefig(buf, format="png", bbox_inches="tight")

            # Get the image bytes
            buf.seek(0)
            image_bytes = buf.getvalue()

            return image_bytes, script
        except Exception as e:
            logging.exception("Error saving plot to memory")
            st.error(f"Error saving plot: {str(e)}")
    else:
        logging.warning("No figure was generated")

    logging.info("Returning script only (no image)")
    return None, script


def main():
    """Main function to render the Streamlit app"""
    st.title("GetDist GUI")

    # Create a menu bar at the top
    menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4)

    # File Menu
    with menu_col1:
        file_menu = st.expander("📁 File")
        with file_menu:
            if st.button(
                "Re-load files", key="reload_button", use_container_width=True, disabled=not st.session_state.chain_dir
            ):
                reload_files()
                st.rerun()

            # Add divider
            st.divider()

            # Add download options to the File menu
            st.markdown("**Export Plot**")

            # Download PNG button
            if st.button(
                "Download PNG",
                key="download_png_button",
                use_container_width=True,
                disabled=not st.session_state.current_plot,
            ):
                with st.spinner("Preparing PNG file..."):
                    # Get the plot data in PNG format
                    file_data, mime_type = export_plot("PNG")
                    if file_data:
                        # Offer the file for download
                        st.download_button(
                            label="Click to download PNG",
                            data=file_data,
                            file_name="getdist_plot.png",
                            mime=mime_type,
                            key="download_png_actual",
                            use_container_width=True,
                        )
                    else:
                        st.error("Failed to generate PNG file")

            # Download PDF button
            if st.button(
                "Download PDF",
                key="download_pdf_button",
                use_container_width=True,
                disabled=not st.session_state.current_plot,
            ):
                with st.spinner("Preparing PDF file..."):
                    # Get the plot data in PDF format
                    file_data, mime_type = export_plot("PDF")
                    if file_data:
                        # Offer the file for download
                        st.download_button(
                            label="Click to download PDF",
                            data=file_data,
                            file_name="getdist_plot.pdf",
                            mime=mime_type,
                            key="download_pdf_actual",
                            use_container_width=True,
                        )
                    else:
                        st.error("Failed to generate PDF file")

    # Helper function to toggle dialogs
    def toggle_dialog(dialog_name, condition=True):
        """Toggle a specific dialog and close all others

        Args:
            dialog_name: The name of the dialog to open
            condition: Optional condition that must be True for the dialog to open

        Returns:
            True if the dialog was opened, False otherwise
        """
        # List of all available dialogs
        all_dialogs = [
            "show_marge_stats",
            "show_like_stats",
            "show_converge_stats",
            "show_pca",
            "show_param_table",
            "show_analysis_settings",
            "show_plot_options",
            "show_config_settings",
            "show_about",
            "show_debug_log",
        ]

        # Close all dialogs
        for dialog in all_dialogs:
            st.session_state[dialog] = False

        # Open the requested dialog if condition is met
        if condition:
            st.session_state[dialog_name] = True
            return True
        return False

    # Data Menu
    with menu_col2:
        data_menu = st.expander("📊 Data")
        with data_menu:
            # Marge Stats button
            if st.button(
                "Marge Stats",
                key="marge_stats_button",
                use_container_width=True,
                disabled=not st.session_state.selected_roots,
            ):
                toggle_dialog("show_marge_stats")

            # Like Stats button
            if st.button(
                "Like Stats",
                key="like_stats_button",
                use_container_width=True,
                disabled=not st.session_state.selected_roots,
            ):
                toggle_dialog("show_like_stats")

            # Converge Stats button
            if st.button(
                "Converge Stats",
                key="converge_stats_button",
                use_container_width=True,
                disabled=not st.session_state.selected_roots,
            ):
                toggle_dialog("show_converge_stats")

            st.divider()

            # Parameter PCA button
            if st.button(
                "Parameter PCA",
                key="pca_button",
                use_container_width=True,
                disabled=not (st.session_state.selected_roots and st.session_state.x_params),
            ):
                toggle_dialog("show_pca")

            # Parameter Table button
            if st.button(
                "Parameter Table",
                key="param_table_button",
                use_container_width=True,
                disabled=not st.session_state.selected_roots,
            ):
                toggle_dialog("show_param_table")

    # Options Menu
    with menu_col3:
        options_menu = st.expander("⚙️ Options")
        with options_menu:
            # Analysis Settings button
            if st.button("Analysis Settings", key="analysis_settings_button", use_container_width=True):
                toggle_dialog("show_analysis_settings")

            # Plot Options button
            if st.button("Plot Options", key="plot_options_button", use_container_width=True):
                toggle_dialog("show_plot_options")

            # Plot style module button
            if st.button("Plot style module", key="config_settings_button", use_container_width=True):
                toggle_dialog("show_config_settings")

            st.divider()

            # Reset buttons (these don't toggle dialogs)
            if st.button("Reset Analysis Settings", key="reset_analysis_button", use_container_width=True):
                reset_analysis_settings()

            if st.button("Reset Plot Options", key="reset_plot_button", use_container_width=True):
                reset_plot_options()
                st.rerun()

    # Help Menu
    with menu_col4:
        help_menu = st.expander("❓ Help")
        with help_menu:
            # Direct links instead of buttons
            st.markdown("[GetDist Documentation](https://getdist.readthedocs.io/)")
            st.markdown("[GetDist on GitHub](https://github.com/cmbant/getdist)")

            st.divider()

            st.markdown("[Planck Legacy Archive](https://pla.esac.esa.int/)")

            st.divider()

            # Debug Log button
            if st.button("View Debug Log", key="debug_log_button", use_container_width=True):
                toggle_dialog("show_debug_log")

            # About button
            if st.button("About", key="about_button", use_container_width=True):
                toggle_dialog("show_about")

    # Handle dialog displays

    if st.session_state.show_marge_stats:
        with st.expander("Marginalized Statistics", expanded=True):
            # Add chain selection
            if len(st.session_state.selected_roots) > 1:
                # Store the previous selection to detect changes
                prev_selection = st.session_state.get("marge_stats_selected_chain", st.session_state.active_chain)

                selected_chain = st.selectbox(
                    "Select chain:",
                    options=st.session_state.selected_roots,
                    index=st.session_state.selected_roots.index(st.session_state.active_chain)
                    if "active_chain" in st.session_state
                    and st.session_state.active_chain in st.session_state.selected_roots
                    else 0,
                    key="marge_stats_chain_select",
                )

                # Update active chain
                st.session_state.active_chain = selected_chain

                # Store the current selection
                st.session_state.marge_stats_selected_chain = selected_chain

                # If selection changed, rerun to refresh the stats
                if prev_selection != selected_chain:
                    st.rerun()
            else:
                selected_chain = st.session_state.selected_roots[0] if st.session_state.selected_roots else None
                st.session_state.active_chain = selected_chain
                st.session_state.marge_stats_selected_chain = selected_chain

            result = show_marge_stats(selected_chain)
            if result:
                rootname, stats = result
                st.subheader(f"Statistics for: {rootname}")

                # Create a table to display the stats
                data = []
                headers = (
                    ["Parameter", "Mean", "Std Dev"]
                    + [f"{lim}% Lower" for lim in [68, 95, 99]]
                    + [f"{lim}% Upper" for lim in [68, 95, 99]]
                    + ["Label"]
                )

                for param in stats.names:
                    row = [param.name, f"{param.mean:.6g}", f"{param.err:.6g}"]
                    for lim in param.limits:
                        row.extend([f"{lim.lower:.6g}", f"{lim.upper:.6g}"])
                    row.append(param.label)
                    data.append(row)

                # Display as a dataframe
                import pandas as pd

                df = pd.DataFrame(data, columns=headers)
                st.dataframe(df, use_container_width=True)

            if st.button("Close", key="close_marge_stats"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_like_stats:
        with st.expander("Likelihood Statistics", expanded=True):
            # Add chain selection
            if len(st.session_state.selected_roots) > 1:
                # Store the previous selection to detect changes
                prev_selection = st.session_state.get("like_stats_selected_chain", st.session_state.active_chain)

                selected_chain = st.selectbox(
                    "Select chain:",
                    options=st.session_state.selected_roots,
                    index=st.session_state.selected_roots.index(st.session_state.active_chain)
                    if "active_chain" in st.session_state
                    and st.session_state.active_chain in st.session_state.selected_roots
                    else 0,
                    key="like_stats_chain_select",
                )

                # Update active chain
                st.session_state.active_chain = selected_chain

                # Store the current selection
                st.session_state.like_stats_selected_chain = selected_chain

                # If selection changed, rerun to refresh the stats
                if prev_selection != selected_chain:
                    st.rerun()
            else:
                selected_chain = st.session_state.selected_roots[0] if st.session_state.selected_roots else None
                st.session_state.active_chain = selected_chain
                st.session_state.like_stats_selected_chain = selected_chain

            result = show_like_stats(selected_chain)
            if result:
                rootname, stats = result
                st.subheader(f"Sample likelihood constraints: {rootname}")

                # Display the likelihood summary
                st.text(stats.likeSummary())

                # Create a table for the detailed stats
                data = []
                headers = ["Parameter", "Best Fit", "68% Lower", "68% Upper", "95% Lower", "95% Upper", "Label"]

                for param in stats.names:
                    row = [param.name]

                    # Add best fit value
                    row.append(f"{param.bestfit_sample:.6g}")

                    # Add confidence limits - hardcoded to match original code
                    # The original code assumes exactly 2 contour levels (68% and 95%)
                    if (
                        hasattr(param, "ND_limit_bot")
                        and hasattr(param, "ND_limit_top")
                        and param.ND_limit_bot.size >= 2
                        and param.ND_limit_top.size >= 2
                    ):
                        row.extend(
                            [
                                f"{param.ND_limit_bot[0]:.6g}",
                                f"{param.ND_limit_top[0]:.6g}",
                                f"{param.ND_limit_bot[1]:.6g}",
                                f"{param.ND_limit_top[1]:.6g}",
                            ]
                        )
                    else:
                        # Fallback if limits not available
                        row.extend(["N/A", "N/A", "N/A", "N/A"])

                    # Add label
                    row.append(param.label)

                    data.append(row)

                # Display as a dataframe
                import pandas as pd

                df = pd.DataFrame(data, columns=headers)
                st.dataframe(df, use_container_width=True)

            if st.button("Close", key="close_like_stats"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_converge_stats:
        with st.expander("Convergence Statistics", expanded=True):
            # Add chain selection
            if len(st.session_state.selected_roots) > 1:
                # Store the previous selection to detect changes
                prev_selection = st.session_state.get("converge_stats_selected_chain", st.session_state.active_chain)

                selected_chain = st.selectbox(
                    "Select chain:",
                    options=st.session_state.selected_roots,
                    index=st.session_state.selected_roots.index(st.session_state.active_chain)
                    if "active_chain" in st.session_state
                    and st.session_state.active_chain in st.session_state.selected_roots
                    else 0,
                    key="converge_stats_chain_select",
                )

                # Update active chain
                st.session_state.active_chain = selected_chain

                # Store the current selection
                st.session_state.converge_stats_selected_chain = selected_chain

                # If selection changed, rerun to refresh the stats
                if prev_selection != selected_chain:
                    st.rerun()
            else:
                selected_chain = st.session_state.selected_roots[0] if st.session_state.selected_roots else None
                st.session_state.active_chain = selected_chain
                st.session_state.converge_stats_selected_chain = selected_chain

            result = show_converge_stats(selected_chain)
            if result:
                rootname, stats, summary = result
                st.subheader(f"Statistics for: {rootname}")

                # Display the summary
                st.text(summary)

                # Create a table for the detailed stats
                data = []
                for param, stat in stats.items():
                    row = [param]
                    row.extend([f"{val:.6g}" for val in stat])
                    data.append(row)

                # Display as a dataframe
                import pandas as pd

                df = pd.DataFrame(
                    data, columns=["Parameter", "R-1", "Var(mean)/mean(var)", "Remaining Chains", "Worst e-value"]
                )
                st.dataframe(df, use_container_width=True)

            if st.button("Close", key="close_converge_stats"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_pca:
        with st.expander("Parameter PCA", expanded=True):
            # Add chain selection
            if len(st.session_state.selected_roots) > 1:
                # Store the previous selection to detect changes
                prev_selection = st.session_state.get("pca_selected_chain", st.session_state.active_chain)

                selected_chain = st.selectbox(
                    "Select chain:",
                    options=st.session_state.selected_roots,
                    index=st.session_state.selected_roots.index(st.session_state.active_chain)
                    if "active_chain" in st.session_state
                    and st.session_state.active_chain in st.session_state.selected_roots
                    else 0,
                    key="pca_chain_select",
                )

                # Update active chain
                st.session_state.active_chain = selected_chain

                # Store the current selection
                st.session_state.pca_selected_chain = selected_chain

                # If selection changed, rerun to refresh the stats
                if prev_selection != selected_chain:
                    st.rerun()
            else:
                selected_chain = st.session_state.selected_roots[0] if st.session_state.selected_roots else None
                st.session_state.active_chain = selected_chain
                st.session_state.pca_selected_chain = selected_chain

            result = show_pca(selected_chain)
            if result:
                rootname, pca_result = result
                st.subheader(f"PCA for: {rootname}")

                # Display PCA results
                st.write("**PCA Results**")

                # Unpack the PCA result
                eigenvalues, eigenvectors, full_pca_text = pca_result
                param_names = st.session_state.x_params

                # Option to show full PCA text output
                if st.checkbox("Show full PCA output", value=False):
                    st.text(full_pca_text)

                # Display eigenvalues
                st.write("**Eigenvalues:**")
                for i, val in enumerate(eigenvalues):
                    st.write(f"PC{i + 1}: {val:.6g}")

                # Display eigenvectors
                st.write("**Eigenvectors:**")
                data = []
                for i, vec in enumerate(eigenvectors):
                    row = [f"PC{i + 1}"] + [f"{v:.6g}" for v in vec]
                    data.append(row)

                # Display as a dataframe
                import pandas as pd

                df = pd.DataFrame(data, columns=["Component"] + param_names)
                st.dataframe(df, use_container_width=True)

            if st.button("Close", key="close_pca"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_param_table:
        with st.expander("Parameter Table", expanded=True):
            if st.session_state.selected_roots:
                # Add chain selection
                if len(st.session_state.selected_roots) > 1:
                    # Store the previous selection to detect changes
                    prev_selection = st.session_state.get("param_table_selected_chain", st.session_state.active_chain)

                    selected_chain = st.selectbox(
                        "Select chain:",
                        options=st.session_state.selected_roots,
                        index=st.session_state.selected_roots.index(st.session_state.active_chain)
                        if "active_chain" in st.session_state
                        and st.session_state.active_chain in st.session_state.selected_roots
                        else 0,
                        key="param_table_chain_select",
                    )

                    # Update active chain
                    st.session_state.active_chain = selected_chain

                    # Store the current selection
                    st.session_state.param_table_selected_chain = selected_chain

                    # If selection changed, rerun to refresh the stats
                    if prev_selection != selected_chain:
                        st.rerun()
                else:
                    selected_chain = st.session_state.selected_roots[0] if st.session_state.selected_roots else None
                    st.session_state.active_chain = selected_chain
                    st.session_state.param_table_selected_chain = selected_chain

                plotter = get_plotter()
                if plotter:
                    samples = plotter.sample_analyser.samples_for_root(selected_chain)
                    if samples and hasattr(samples, "paramNames"):
                        st.subheader(f"Parameters for: {selected_chain}")

                        # Create tabs for different table types (like in the original)
                        tab1, tab2 = st.tabs(["Parameter List", "Parameter Tables"])

                        with tab1:
                            # Create a table of all parameters
                            data = []
                            for param in samples.paramNames.names:
                                row = [param.name, param.label]
                                data.append(row)

                            # Display as a dataframe
                            import pandas as pd

                            df = pd.DataFrame(data, columns=["Parameter", "Label"])
                            st.dataframe(df, use_container_width=True)

                        with tab2:
                            # Get parameter tables for different confidence levels
                            try:
                                # Get selected parameters or all parameters if none selected
                                pars = st.session_state.x_params
                                ignore_unknown = False
                                if len(pars) < 1:
                                    pars = [param.name for param in samples.paramNames.names]
                                    # If no parameters selected, it shouldn't fail if some sample is missing
                                    # parameters present in the first one
                                    ignore_unknown = True

                                # Add renames to match parameter across samples
                                renames = samples.paramNames.getRenames(keep_empty=True)
                                pars = [
                                    getattr(
                                        samples.paramNames.parWithName(p, error=not ignore_unknown, renames=renames),
                                        "name",
                                        None,
                                    )
                                    for p in pars
                                ]
                                # Remove None values
                                pars = [p for p in pars if p is not None]

                                if len(pars) > 0:
                                    # Create tables for different confidence levels
                                    tables = [
                                        samples.getTable(columns=len(pars) // 20 + 1, limit=lim + 1, paramList=pars)
                                        for lim in range(len(samples.contours))
                                    ]

                                    # Create tabs for different confidence levels and display options
                                    conf_tabs = st.tabs(
                                        [
                                            f"{samples.contours[i] * 100:.0f}% limits"
                                            for i in range(len(samples.contours))
                                        ]
                                    )

                                    for i, tab in enumerate(conf_tabs):
                                        with tab:
                                            # Add option to switch between LaTeX and DataFrame view
                                            view_type = st.radio(
                                                "View as:",
                                                options=["LaTeX", "Table"],
                                                horizontal=True,
                                                key=f"view_type_{i}",
                                                help="LaTeX view shows the raw LaTeX code that can be copied for use in papers. Table view shows a formatted table.",
                                            )

                                            if view_type == "Table":
                                                # Create a more readable table using pandas DataFrame
                                                try:
                                                    # Extract table data from the getdist table object
                                                    table = tables[i]

                                                    # Create a DataFrame with the table data
                                                    import pandas as pd

                                                    # Extract data from the ResultTable object
                                                    # ResultTable doesn't have blockDat attribute, so we need to parse the LaTeX output
                                                    param_data = []

                                                    # Get the table content as LaTeX
                                                    latex_text = table.tableTex()

                                                    # Find the tabular environment
                                                    tabular_match = re.search(
                                                        r"\\begin\{tabular\}.*?\\end\{tabular\}", latex_text, re.DOTALL
                                                    )
                                                    if tabular_match:
                                                        tabular_content = tabular_match.group(0)

                                                        # Split into rows (split on \\)
                                                        rows = re.split(r"\\\\", tabular_content)

                                                        # Skip header rows and process data rows
                                                        # Usually first few rows are headers and formatting
                                                        data_rows = [
                                                            row
                                                            for row in rows
                                                            if "&" in row
                                                            and "\\multicolumn" not in row
                                                            and "\\hline" not in row
                                                        ]

                                                        # Process each data row
                                                        for row in data_rows:
                                                            # Split into cells (split on &)
                                                            cells = row.split("&")
                                                            if len(cells) >= 2:  # At least parameter name and one value
                                                                # Clean up cells
                                                                clean_cells = []
                                                                for cell in cells:
                                                                    # Remove LaTeX formatting
                                                                    cell = re.sub(
                                                                        r"\\[a-zA-Z]+\{([^\}]*)\}", r"\1", cell
                                                                    )
                                                                    cell = cell.strip()
                                                                    # Remove $ signs from math mode
                                                                    cell = cell.replace("$", "")
                                                                    clean_cells.append(cell)

                                                                # First cell is parameter name
                                                                param_data.append(clean_cells)

                                                    # Create column headers based on the extracted data
                                                    # First, determine how many columns we have in our data
                                                    if param_data:
                                                        num_cols = len(param_data[0])

                                                        # Create appropriate column headers
                                                        if num_cols == 2:  # Just parameter name and value
                                                            columns = ["Parameter", "Value"]
                                                        elif num_cols == 3:  # Parameter name, mean, and std dev
                                                            columns = ["Parameter", "Mean", "Std Dev"]
                                                        elif (
                                                            num_cols == 5
                                                        ):  # Parameter name, mean, std dev, lower, upper
                                                            if i == 0:  # 68% limits
                                                                columns = [
                                                                    "Parameter",
                                                                    "Mean",
                                                                    "Std Dev",
                                                                    "Lower",
                                                                    "Upper",
                                                                ]
                                                            else:  # 95% or 99% limits
                                                                columns = [
                                                                    "Parameter",
                                                                    "Mean",
                                                                    "Std Dev",
                                                                    f"{samples.contours[i] * 100:.0f}% Lower",
                                                                    f"{samples.contours[i] * 100:.0f}% Upper",
                                                                ]
                                                        else:  # Generic column headers
                                                            columns = [f"Column {j + 1}" for j in range(num_cols)]
                                                            columns[0] = (
                                                                "Parameter"  # First column is always parameter name
                                                            )

                                                        # Create the DataFrame
                                                        df = pd.DataFrame(param_data, columns=columns)
                                                    else:
                                                        # If no data was extracted, create an empty DataFrame
                                                        df = pd.DataFrame(columns=["Parameter", "Value"])
                                                        st.warning(
                                                            "No parameter data could be extracted from the table."
                                                        )

                                                    # Display the table
                                                    st.dataframe(df, use_container_width=True)
                                                except Exception as df_err:
                                                    st.error(f"Error creating table view: {str(df_err)}")
                                                    st.info("Falling back to LaTeX view")
                                                    view_type = "LaTeX"

                                            if view_type == "LaTeX":
                                                # Get the table as text
                                                table_text = tables[i].tableTex()

                                                # Display raw LaTeX with a copy button
                                                st.write("### LaTeX Table")
                                                st.info(
                                                    "The LaTeX table is shown in raw format below. Use the copy button to copy it to your clipboard."
                                                )

                                                # Display the raw LaTeX in a text area
                                                st.text_area(
                                                    "LaTeX Code", table_text, height=300, key=f"latex_text_{i}"
                                                )

                                                # Add copy button
                                                if st.button("Copy LaTeX", key=f"copy_latex_{i}"):
                                                    st.session_state[f"clipboard_{i}"] = table_text
                                                    st.success("LaTeX copied to clipboard!")
                            except Exception as e:
                                st.error(f"Error generating parameter tables: {str(e)}")

            if st.button("Close", key="close_param_table"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_analysis_settings:
        with st.expander("Analysis Settings", expanded=True):
            # Create a form for analysis settings
            with st.form("analysis_settings_form"):
                settings = {}

                default_names = IniFile(getdist.default_getdist_settings)

                # Get items in the same order as the original code
                items = []

                # First add items in the read order from default settings
                for key in default_names.readOrder:
                    if key in st.session_state.current_settings.params:
                        items.append(key)

                # Then add any remaining items
                for key in st.session_state.current_settings.params:
                    if key not in items and key in default_names.params:
                        items.append(key)

                # Display all settings in a table format like the original
                st.subheader("Analysis Settings")

                # Create a compact table-like layout
                # Group parameters into chunks for multi-column display
                chunk_size = 10  # Number of parameters per column group
                item_chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

                for chunk in item_chunks:
                    # Create columns for each group
                    cols = st.columns(2)

                    # First column: first half of the chunk
                    with cols[0]:
                        for key in chunk[: len(chunk) // 2 + len(chunk) % 2]:
                            value = st.session_state.current_settings.string(key)
                            is_bool = value in ["False", "True"]

                            # Parameter name and value in a compact layout
                            st.markdown(f"**{key}**")
                            if is_bool:
                                settings[key] = str(
                                    st.checkbox(
                                        key,
                                        value=st.session_state.current_settings.bool(key),
                                        key=f"checkbox_{key}",
                                        help=f"Parameter: {key}",
                                        label_visibility="collapsed",
                                    )
                                )
                            else:
                                settings[key] = st.text_input(
                                    key,
                                    value=value,
                                    key=f"input_{key}",
                                    label_visibility="collapsed",
                                    help=f"Parameter: {key}",
                                )

                    # Second column: second half of the chunk
                    with cols[1]:
                        for key in chunk[len(chunk) // 2 + len(chunk) % 2 :]:
                            value = st.session_state.current_settings.string(key)
                            is_bool = value in ["False", "True"]

                            # Parameter name and value in a compact layout
                            st.markdown(f"**{key}**")
                            if is_bool:
                                settings[key] = str(
                                    st.checkbox(
                                        key,
                                        value=st.session_state.current_settings.bool(key),
                                        key=f"checkbox_{key}",
                                        help=f"Parameter: {key}",
                                        label_visibility="collapsed",
                                    )
                                )
                            else:
                                settings[key] = st.text_input(
                                    key,
                                    value=value,
                                    key=f"input_{key}",
                                    label_visibility="collapsed",
                                    help=f"Parameter: {key}",
                                )

                # Submit button
                submitted = st.form_submit_button("Update")
                if submitted:
                    apply_analysis_settings(settings)

            if st.button("Close", key="close_analysis_settings"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_plot_options:
        with st.expander("Plot Settings", expanded=True):
            # Create a form for plot options
            with st.form("plot_options_form"):
                # Get the default plot settings from the plotter
                if not hasattr(st.session_state, "default_plot_settings") and st.session_state.plotter:
                    st.session_state.default_plot_settings = copy.copy(st.session_state.plotter.settings)
                    st.session_state.custom_plot_settings = {}

                # Get plot settings documentation
                if hasattr(st.session_state, "default_plot_settings"):
                    settings = st.session_state.default_plot_settings
                    pars = []
                    skips = ["param_names_for_labels", "progress"]
                    comments = {}

                    # Extract parameters and comments from docstring
                    if hasattr(settings, "__doc__") and settings.__doc__:
                        for line in settings.__doc__.split("\n"):
                            if "ivar" in line:
                                try:
                                    items = line.split(":", 2)
                                    par = items[1].split("ivar ")[1]
                                    if par not in skips:
                                        pars.append(par)
                                        comments[par] = items[2].strip() if len(items) > 2 else ""
                                except Exception:
                                    pass

                    # Sort parameters
                    pars.sort()

                    # Create a dictionary to store settings
                    settings_dict = {}

                    # Display all settings in a table-like layout
                    st.subheader("Plot Settings")

                    # Create a compact table-like layout
                    # Group parameters into chunks for multi-column display
                    chunk_size = 10  # Number of parameters per column group
                    par_chunks = [pars[i : i + chunk_size] for i in range(0, len(pars), chunk_size)]

                    for chunk in par_chunks:
                        # Create columns for each group
                        cols = st.columns(2)

                        # First column: first half of the chunk
                        with cols[0]:
                            for par in chunk[: len(chunk) // 2 + len(chunk) % 2]:
                                # Get current value
                                if par in st.session_state.custom_plot_settings:
                                    current_value = st.session_state.custom_plot_settings[par]
                                elif hasattr(settings, par):
                                    current_value = getattr(settings, par)
                                else:
                                    current_value = ""

                                # Convert to string for display
                                if current_value is None:
                                    value_str = ""
                                elif isinstance(current_value, bool):
                                    value_str = str(current_value)
                                else:
                                    value_str = str(current_value)

                                # Parameter name and value in a compact layout
                                st.markdown(f"**{par}**")
                                is_bool = value_str.lower() in ["true", "false"]
                                if is_bool:
                                    settings_dict[par] = str(
                                        st.checkbox(
                                            par,
                                            value=value_str.lower() == "true",
                                            key=f"plot_checkbox_{par}_1",
                                            label_visibility="collapsed",
                                            help=comments.get(par, ""),
                                        )
                                    )
                                else:
                                    settings_dict[par] = st.text_input(
                                        par,
                                        value=value_str,
                                        key=f"plot_input_{par}_1",
                                        label_visibility="collapsed",
                                        help=comments.get(par, ""),
                                    )

                        # Second column: second half of the chunk
                        with cols[1]:
                            for par in chunk[len(chunk) // 2 + len(chunk) % 2 :]:
                                # Get current value
                                if par in st.session_state.custom_plot_settings:
                                    current_value = st.session_state.custom_plot_settings[par]
                                elif hasattr(settings, par):
                                    current_value = getattr(settings, par)
                                else:
                                    current_value = ""

                                # Convert to string for display
                                if current_value is None:
                                    value_str = ""
                                elif isinstance(current_value, bool):
                                    value_str = str(current_value)
                                else:
                                    value_str = str(current_value)

                                # Parameter name and value in a compact layout
                                st.markdown(f"**{par}**")
                                is_bool = value_str.lower() in ["true", "false"]
                                if is_bool:
                                    settings_dict[par] = str(
                                        st.checkbox(
                                            par,
                                            value=value_str.lower() == "true",
                                            key=f"plot_checkbox_{par}",
                                            label_visibility="collapsed",
                                            help=comments.get(par, ""),
                                        )
                                    )
                                else:
                                    settings_dict[par] = st.text_input(
                                        par,
                                        value=value_str,
                                        key=f"plot_input_{par}",
                                        label_visibility="collapsed",
                                        help=comments.get(par, ""),
                                    )

                    # Submit button
                    submitted = st.form_submit_button("Update")
                    if submitted:
                        # Process settings
                        deleted = []
                        try:
                            st.session_state.custom_plot_settings = {}
                            for key, value in settings_dict.items():
                                if hasattr(settings, key):
                                    current = getattr(settings, key)
                                    if str(current) != value and len(value):
                                        if isinstance(current, str):
                                            st.session_state.custom_plot_settings[key] = value
                                        else:
                                            try:
                                                st.session_state.custom_plot_settings[key] = eval(value)
                                            except Exception:
                                                if current is None or re.match(r"^[\w]+$", value):
                                                    st.session_state.custom_plot_settings[key] = value
                                                else:
                                                    raise Exception(f"Invalid value for {key}: {value}")
                                    else:
                                        deleted.append(key)
                                        st.session_state.custom_plot_settings.pop(key, None)

                            # Update plot settings
                            st.session_state.plot_settings = {}
                            for key, value in st.session_state.custom_plot_settings.items():
                                st.session_state.plot_settings[key] = value

                            st.success("Plot settings updated")
                        except Exception as e:
                            st.error(f"Error updating plot settings: {str(e)}")
                else:
                    st.warning("No plotter available. Open a chain directory first.")

            if st.button("Close", key="close_plot_options"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_config_settings:
        with st.expander("Plot Style Module", expanded=True):
            # Create a form for config settings
            with st.form("config_settings_form"):
                # Plot module settings
                st.subheader("Plot Style Module")
                plot_module = st.text_input("Plot Module", value=st.session_state.plot_module)
                script_plot_module = st.text_input("Script Plot Module", value=st.session_state.script_plot_module)

                # Submit button
                submitted = st.form_submit_button("Update")
                if submitted:
                    # Update settings
                    st.session_state.plot_module = plot_module
                    st.session_state.script_plot_module = script_plot_module
                    apply_plot_module(plot_module)
                    st.success("Plot style module updated")

            if st.button("Close", key="close_config_settings"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_about:
        with st.expander("About GetDist GUI", expanded=True):
            st.write("**GetDist GUI**")
            st.write("A graphical user interface for the GetDist package.")
            st.write(
                "GetDist is a Python package for analysing Monte Carlo samples, including correlated samples from Markov Chain Monte Carlo (MCMC)."
            )
            st.write("\nDeveloped by Antony Lewis and contributors.")
            st.write("\nStreamlit version by Augment Code.")
            st.write("\n[GetDist Documentation](https://getdist.readthedocs.io/)")
            st.write("[GetDist on GitHub](https://github.com/cmbant/getdist)")

            if st.button("Close", key="close_about"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    if st.session_state.show_debug_log:
        with st.expander("Debug Log", expanded=True):
            # Get the log file path
            log_file_path = os.path.join(os.path.dirname(__file__), "getdist_streamlit.log")

            if os.path.exists(log_file_path):
                try:
                    # Read the log file
                    with open(log_file_path) as f:
                        log_content = f.read()

                    # Add refresh button and log file path info
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"Log file: {log_file_path}")
                    with col2:
                        if st.button("Refresh", key="refresh_log"):
                            st.rerun()

                    # Show the log content in a text area with scrolling
                    st.text_area("Log Content", log_content, height=400, key="log_content")

                    # Add download button for the log file
                    if st.download_button(
                        label="Download Log File",
                        data=log_content,
                        file_name="getdist_streamlit.log",
                        mime="text/plain",
                        key="download_log",
                    ):
                        pass
                except Exception as e:
                    st.error(f"Error reading log file: {str(e)}")
            else:
                st.warning(f"Log file not found at: {log_file_path}")

            if st.button("Close", key="close_debug_log"):
                toggle_dialog(None)  # Close all dialogs
                st.rerun()

    # Create a sidebar for controls
    with st.sidebar:
        # Directory selection
        st.subheader("Chain Directory")

        # Recent directories and browser path are initialized at the top of the script

        # Create a compact layout for directory selection
        dir_col1, dir_col2 = st.columns([5, 1])

        # Variable to store the directory path
        dir_path = None

        # Different UI based on whether a directory is already open
        if st.session_state.chain_dir:
            # When a directory is already open, only show the dropdown
            with dir_col1:
                if st.session_state.recent_directories:
                    # Use a container to reduce vertical spacing
                    with st.container():
                        selected_dir = st.selectbox(
                            "Recent directories",
                            options=st.session_state.recent_directories,
                            key="recent_dir_select",
                            label_visibility="collapsed",
                        )
                        if selected_dir:
                            dir_path = selected_dir

            with dir_col2:
                # Browse button with folder icon
                if st.button("📂", key="browse_button", help="Browse for directory"):
                    st.session_state.show_file_browser = True
                    st.rerun()

            # Change directory button
            if dir_path and dir_path != st.session_state.chain_dir:
                if st.button("Change Directory", key="change_dir_button", use_container_width=True):
                    # Show a loading spinner while opening the directory
                    with st.spinner(f"Opening directory: {dir_path}..."):
                        if open_directory(dir_path):
                            # Add to recent directories if successful
                            if dir_path not in st.session_state.recent_directories:
                                st.session_state.recent_directories.insert(0, dir_path)
                                # Keep only the 10 most recent
                                if len(st.session_state.recent_directories) > 10:
                                    st.session_state.recent_directories = st.session_state.recent_directories[:10]
                                # Save the updated list to file
                                save_recent_directories()
                            # Force a rerun to update the UI
                            st.rerun()
        else:
            # When no directory is open, show both dropdown and text input
            with dir_col1:
                # Dropdown for recent directories with auto-fill
                if st.session_state.recent_directories:
                    # Use a container to reduce vertical spacing
                    with st.container():
                        selected_dir = st.selectbox(
                            "Recent directories",
                            options=st.session_state.recent_directories,
                            key="recent_dir_select",
                            label_visibility="collapsed",
                        )
                        # Auto-fill the text input when selecting from dropdown
                        if selected_dir:
                            # Store the selected directory to use as default value for the input field
                            st.session_state.display_dir_path = selected_dir

                # Initialize dir_path_input in session state if needed
                if "display_dir_path" in st.session_state and st.session_state.display_dir_path is not None:
                    # Update the session state value instead of the widget value directly
                    st.session_state.dir_path_input = st.session_state.display_dir_path
                    # Reset display_dir_path after using it
                    st.session_state.display_dir_path = None

                # Text input for directory path
                dir_path = st.text_input(
                    "Directory path",
                    key="dir_path_input",
                    label_visibility="collapsed",
                    placeholder="Enter directory path",
                )

            with dir_col2:
                # Browse button with folder icon
                if st.button("📂", key="browse_button", help="Browse for directory"):
                    st.session_state.show_file_browser = True
                    st.rerun()

            # Open directory button
            if st.button("Open Directory", key="open_dir_button", use_container_width=True):
                if dir_path:
                    # Show a loading spinner while opening the directory
                    with st.spinner(f"Opening directory: {dir_path}..."):
                        if open_directory(dir_path):
                            # Add to recent directories if successful
                            if dir_path not in st.session_state.recent_directories:
                                st.session_state.recent_directories.insert(0, dir_path)
                                # Keep only the 10 most recent
                                if len(st.session_state.recent_directories) > 10:
                                    st.session_state.recent_directories = st.session_state.recent_directories[:10]
                                # Save the updated list to file
                                save_recent_directories()
                            # Force a rerun to update the UI
                            st.rerun()

        # File browser dialog is initialized at the top of the script
        # Check if a directory was selected in the previous run
        if st.session_state.selected_directory:
            # Use the selected directory
            dir_path = st.session_state.selected_directory
            # Reset the selected directory
            st.session_state.selected_directory = None

            # Show a loading spinner while opening the directory
            with st.spinner(f"Opening directory: {dir_path}..."):
                # Open the directory
                if open_directory(dir_path):
                    # Add to recent directories if successful
                    if dir_path not in st.session_state.recent_directories:
                        st.session_state.recent_directories.insert(0, dir_path)
                        # Keep only the 10 most recent
                        if len(st.session_state.recent_directories) > 10:
                            st.session_state.recent_directories = st.session_state.recent_directories[:10]
                        # Save the updated list to file
                        save_recent_directories()
                    # Force a rerun to update the UI
                    st.rerun()

        # Show file browser if requested
        if st.session_state.show_file_browser:
            with st.expander("Select Directory", expanded=True):
                # Show current path (read-only)
                st.text(f"Current path: {st.session_state.current_browser_path}")

                # Get directories in the current path
                try:
                    dirs = [
                        d
                        for d in os.listdir(st.session_state.current_browser_path)
                        if os.path.isdir(os.path.join(st.session_state.current_browser_path, d))
                    ]
                    dirs.sort()

                    # Add parent directory option
                    dirs.insert(0, "..")

                    # Create a grid of directory buttons (3 columns)
                    cols = st.columns(3)
                    for i, d in enumerate(dirs):
                        with cols[i % 3]:
                            if st.button(f"📁 {d}", key=f"dir_{d}"):
                                # Handle directory navigation
                                if d == "..":
                                    # Go up one level
                                    st.session_state.current_browser_path = os.path.dirname(
                                        st.session_state.current_browser_path
                                    )
                                else:
                                    # Go into selected directory
                                    st.session_state.current_browser_path = os.path.join(
                                        st.session_state.current_browser_path, d
                                    )
                                st.rerun()
                except Exception as e:
                    st.error(f"Error accessing directory: {str(e)}")

                # Buttons to select or cancel
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select This Directory", key="select_dir_button"):
                        # Store the selected directory in session state
                        st.session_state.selected_directory = st.session_state.current_browser_path
                        st.session_state.show_file_browser = False
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_browser_button"):
                        st.session_state.show_file_browser = False
                        st.rerun()

        # Chain selection
        if st.session_state.chain_dir:
            # If we have a batch with base_dir_names
            if (
                st.session_state.batch
                and hasattr(st.session_state.batch, "base_dir_names")
                and st.session_state.batch.base_dir_names
            ):
                # Parameter tag selection (base directories)
                param_tags = sorted(st.session_state.batch.base_dir_names)

                # Create a dropdown for base directories
                st.write("**Base Directory:**")
                selected_param_tag = st.selectbox(
                    "Select base directory", options=param_tags, key="param_tag_select", label_visibility="collapsed"
                )

                # selected_chain_to_add is initialized at the top of the script

                if selected_param_tag:
                    # Get data tags for this parameter tag
                    data_tags = []
                    try:
                        if isinstance(st.session_state.batch, ChainDirGrid):
                            data_tags = st.session_state.batch.roots_for_dir(selected_param_tag)
                    except Exception as e:
                        st.error(f"Error getting chains for {selected_param_tag}: {str(e)}")

                    # Data tag selection (chains)
                    if data_tags:
                        # Create a dropdown for chains with a compact label
                        st.write("**Chains:**")
                        # Add an empty option to allow no selection by default
                        chain_options = ["Select a chain..."] + data_tags
                        selected_data_tag = st.selectbox(
                            "Select chain to add",
                            options=chain_options,
                            key="data_tag_select",
                            label_visibility="collapsed",
                        )
                        # Convert the placeholder back to None
                        if selected_data_tag == "Select a chain...":
                            selected_data_tag = None

                        # Check if selection changed
                        if selected_data_tag != st.session_state.previous_chain_selection:
                            # Store the new selection
                            st.session_state.selected_chain_to_add = selected_data_tag
                            st.session_state.previous_chain_selection = selected_data_tag

                            # Add the chain automatically if it's not already added
                            if selected_data_tag and selected_data_tag not in st.session_state.selected_roots:
                                # Show a loading spinner while adding the chain
                                with st.spinner(f"Adding chain: {selected_data_tag}..."):
                                    add_root(selected_data_tag)
                                st.success(f"Added chain: {selected_data_tag}")

            # If we have a regular directory with chains
            elif st.session_state.chain_dir and not st.session_state.batch:
                # Get chain files
                root_list = get_chain_root_files(st.session_state.chain_dir)

                # selected_chain_to_add is initialized at the top of the script

                if root_list:
                    st.write("**Available Chains:**")
                    # Create a dropdown for chains
                    # Add an empty option to allow no selection by default
                    chain_options = ["Select a chain..."] + root_list
                    selected_root = st.selectbox(
                        "Select chain to add", options=chain_options, key="root_select", label_visibility="collapsed"
                    )
                    # Convert the placeholder back to None
                    if selected_root == "Select a chain...":
                        selected_root = None

                    # Check if selection changed
                    if selected_root != st.session_state.previous_chain_selection:
                        # Store the new selection
                        st.session_state.selected_chain_to_add = selected_root
                        st.session_state.previous_chain_selection = selected_root

                        # Add the chain automatically if it's not already added
                        if selected_root and selected_root not in st.session_state.selected_roots:
                            # Show a loading spinner while adding the chain
                            with st.spinner(f"Adding chain: {selected_root}..."):
                                add_root(selected_root)
                            st.success(f"Added chain: {selected_root}")

            # Show selected chains
            if st.session_state.selected_roots:
                st.subheader("Selected Chains")

                # Set active chain if not set
                if (
                    not st.session_state.active_chain
                    or st.session_state.active_chain not in st.session_state.selected_roots
                ):
                    st.session_state.active_chain = st.session_state.selected_roots[0]

                # Create a compact container for selected chains
                with st.container(border=True):
                    # Create a more compact list layout
                    chain_cols = st.columns([1, 7, 1])

                    # Add a Remove button at the top
                    with chain_cols[2]:
                        if st.button(
                            "✕",
                            key="remove_selected",
                            help="Remove selected chain",
                            disabled=not st.session_state.selected_roots,
                        ):
                            # Get the currently selected chain
                            if st.session_state.active_chain in st.session_state.selected_roots:
                                # Get the plotter to remove the chain from the sample analyzer
                                plotter = get_plotter()
                                if plotter:
                                    try:
                                        # Remove the chain from the plotter's sample analyzer
                                        plotter.sample_analyser.remove_root(st.session_state.active_chain)
                                        # Remove from root_infos
                                        st.session_state.root_infos.pop(st.session_state.active_chain, None)
                                        logging.info(f"Removed chain {st.session_state.active_chain} from plotter")
                                    except Exception as e:
                                        logging.error(f"Error removing chain from plotter: {str(e)}")

                                # Remove the active chain from selected roots
                                st.session_state.selected_roots.remove(st.session_state.active_chain)

                                # Update active chain if needed
                                if st.session_state.selected_roots:
                                    st.session_state.active_chain = st.session_state.selected_roots[0]
                                    # Update parameters after removing a chain
                                    update_parameters()
                                else:
                                    st.session_state.active_chain = None
                                    st.session_state.param_names = (
                                        None  # Clear parameter names when all chains are removed
                                    )
                                    st.session_state.x_params = []
                                    st.session_state.y_params = []

                                # Force rerun to update UI
                                st.rerun()

                    # Create a list of chains
                    for i, root in enumerate(st.session_state.selected_roots):
                        # Use a single row with minimal height
                        with chain_cols[0]:
                            # Highlight the active chain with an arrow
                            if root == st.session_state.active_chain:
                                st.write("➤")
                            else:
                                st.write(f"{i + 1}")

                        with chain_cols[1]:
                            # Make the chain name clickable to set as active
                            # Use a button that looks like a link
                            if st.button(root, key=f"select_{i}", use_container_width=True):
                                st.session_state.active_chain = root
                                st.rerun()

                    # Add compact reordering controls if we have multiple chains
                    if len(st.session_state.selected_roots) > 1:
                        # Get index of active chain
                        active_index = st.session_state.selected_roots.index(st.session_state.active_chain)

                        # Create a row for reordering buttons
                        reorder_cols = st.columns([3, 1, 1])

                        with reorder_cols[0]:
                            st.write("Reorder active chain:")

                        with reorder_cols[1]:
                            # Move up button (arrow up symbol)
                            if st.button("↑", key="move_up", disabled=(active_index == 0), help="Move up"):
                                # Swap with previous item
                                idx = active_index
                                st.session_state.selected_roots[idx], st.session_state.selected_roots[idx - 1] = (
                                    st.session_state.selected_roots[idx - 1],
                                    st.session_state.selected_roots[idx],
                                )
                                # Update parameters after changing order
                                update_parameters()
                                st.rerun()

                        with reorder_cols[2]:
                            # Move down button (arrow down symbol)
                            if st.button(
                                "↓",
                                key="move_down",
                                disabled=(active_index == len(st.session_state.selected_roots) - 1),
                                help="Move down",
                            ):
                                # Swap with next item
                                idx = active_index
                                st.session_state.selected_roots[idx], st.session_state.selected_roots[idx + 1] = (
                                    st.session_state.selected_roots[idx + 1],
                                    st.session_state.selected_roots[idx],
                                )
                                # Update parameters after changing order
                                update_parameters()
                                st.rerun()

                # Add a button to clear all selected chains
                if st.button("Clear All Selected Chains", key="clear_all_chains"):
                    st.session_state.selected_roots = []
                    st.session_state.active_chain = None
                    st.session_state.param_names = None  # Clear parameter names when all chains are removed
                    st.session_state.x_params = []
                    st.session_state.y_params = []
                    st.rerun()

            # Parameter selection
            if st.session_state.selected_roots and st.session_state.param_names:
                # Get parameter list
                param_list = []
                try:
                    if hasattr(st.session_state.param_names, "list"):
                        param_list = st.session_state.param_names.list()
                    elif hasattr(st.session_state.param_names, "names"):
                        # Fallback if list() method is not available
                        param_list = [p.name for p in st.session_state.param_names.names]
                    else:
                        st.warning("Parameter names object doesn't have expected attributes")
                        # Try to get parameters from the first chain directly
                        if st.session_state.selected_roots:
                            plotter = get_plotter()
                            if plotter:
                                samples = plotter.sample_analyser.samples_for_root(st.session_state.selected_roots[0])
                                if samples and hasattr(samples, "getParamNames"):
                                    param_names = samples.getParamNames()
                                    if hasattr(param_names, "list"):
                                        param_list = param_names.list()
                                    elif hasattr(param_names, "names"):
                                        param_list = [p.name for p in param_names.names]
                except Exception as e:
                    st.error(f"Error getting parameter list: {str(e)}")
                    logging.exception("Parameter list error")

                # Parameter selection section

                # Create a container with a border for parameter selection
                with st.container(border=True):
                    # Create a simple table layout
                    # Initialize parameter selections
                    x_selections = {}
                    y_selections = {}

                    # Create a simple table with three columns
                    col1, col2, col3 = st.columns([8, 1, 1])

                    # Add headers
                    with col1:
                        st.write("Parameter")
                    with col2:
                        st.write("X")
                    with col3:
                        st.write("Y")

                    # Add a separator line
                    st.markdown(
                        "<hr style='margin: 0; border-color: rgba(250, 250, 250, 0.2);'>", unsafe_allow_html=True
                    )

                    # Display parameters in original order with X/Y checkboxes
                    for param in param_list:
                        is_x_selected = param in st.session_state.x_params
                        is_y_selected = param in st.session_state.y_params

                        # Create a row with three columns for each parameter
                        row_cols = st.columns([8, 1, 1])

                        with row_cols[0]:
                            # Use markdown instead of write for more compact display
                            st.markdown(f"<div style='padding: 0; margin: 0;'>{param}</div>", unsafe_allow_html=True)
                        with row_cols[1]:
                            x_selections[param] = st.checkbox(
                                "X", value=is_x_selected, key=f"x_{param}", label_visibility="collapsed"
                            )
                        with row_cols[2]:
                            y_selections[param] = st.checkbox(
                                "Y", value=is_y_selected, key=f"y_{param}", label_visibility="collapsed"
                            )

                        # Add a very thin separator between rows to guide the eye
                        st.markdown(
                            "<hr style='margin: 0; padding: 0; border-color: rgba(250, 250, 250, 0.1); border-width: 1px;'>",
                            unsafe_allow_html=True,
                        )

                    # No need to extract selections - they're already in x_selections and y_selections

                # Create lists of selected parameters
                x_params = [param for param in param_list if x_selections.get(param, False)]
                y_params = [param for param in param_list if y_selections.get(param, False)]

                # Show selected parameters
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Selected X Parameters:**")
                    if x_params:
                        for param in x_params:
                            st.write(f"- {param}")
                    else:
                        st.write("*None selected*")

                with col2:
                    st.write("**Selected Y Parameters:**")
                    if y_params:
                        for param in y_params:
                            st.write(f"- {param}")
                    else:
                        st.write("*None selected*")

                # Update session state
                if x_params != st.session_state.x_params:
                    st.session_state.x_params = x_params

                if y_params != st.session_state.y_params:
                    st.session_state.y_params = y_params

            # Plot type selection
            if st.session_state.selected_roots and st.session_state.x_params:
                # Use a simple radio button for plot type selection
                plot_type = st.radio(
                    "Select plot type:",
                    options=["1D Density", "2D Contour", "Triangle"],
                    index=["1D Density", "2D Contour", "Triangle"].index(st.session_state.plot_type)
                    if st.session_state.plot_type in ["1D Density", "2D Contour", "Triangle"]
                    else 0,
                    horizontal=True,
                )

                if plot_type != st.session_state.plot_type:
                    st.session_state.plot_type = plot_type
                    st.rerun()

                # Generate plot button
                if st.button("Generate Plot", key="generate_plot_button", use_container_width=True):
                    with st.spinner("Generating plot..."):
                        logging.info("Generate Plot button clicked")
                        image_bytes, script = generate_plot()
                        if image_bytes:
                            logging.info("Setting current_plot in session state")
                            st.session_state.current_plot = image_bytes
                            if script:
                                logging.info("Setting current_script in session state")
                                st.session_state.current_script = script
                            st.success("Plot generated!")
                            st.rerun()

    # Main content area - use a two-column layout
    if st.session_state.chain_dir:
        # Create a main column for the plot and a sidebar for settings
        main_col, settings_col = st.columns([3, 1])

        with main_col:
            # Create tabs for Plot and Script
            tab1, tab3 = st.tabs(["Plot", "Script"])

        # Plot Settings in the settings column
        with settings_col:
            # Add Ignore rows option (same as in Analysis Settings)
            current_ignore_rows = st.session_state.current_settings.string("ignore_rows")

            # Display the ignore_rows setting
            ignore_rows_value = st.text_input(
                "Ignore rows",
                value=current_ignore_rows,
                help="Burn in: number of rows to ignore (if > 1), or fraction of rows to ignore (if < 1)",
                key="settings_col_ignore_rows",
            )

            # Store the value in session state to be used when generating the plot
            st.session_state.plot_settings["ignore_rows"] = ignore_rows_value

            # Check if the value has changed and apply it to analysis settings
            if ignore_rows_value != current_ignore_rows:
                # Apply the new value to analysis settings
                apply_analysis_settings({"ignore_rows": ignore_rows_value})

            # Get all available parameters for coloring (like in the original code)
            color_params = ["None"]
            if st.session_state.param_names and hasattr(st.session_state.param_names, "list"):
                color_params.extend(st.session_state.param_names.list())

            # Settings specific to plot types
            st.subheader(f"Settings for {st.session_state.plot_type}")

            if st.session_state.plot_type == "1D Density":
                st.session_state.plot_settings["normalized"] = st.checkbox(
                    "Normalize", value=st.session_state.plot_settings.get("normalized", True)
                )

                # Set color_by to None for 1D plots
                st.session_state.plot_settings["color_by"] = "None"

            elif st.session_state.plot_type == "2D Contour":
                # Use radio buttons for plot type to match original getdist GUI
                plot_type_options = ["Filled", "Line"]
                plot_type_index = 0 if st.session_state.plot_settings.get("filled", True) else 1
                plot_type = st.radio("Plot Type", options=plot_type_options, index=plot_type_index, horizontal=True)
                st.session_state.plot_settings["filled"] = plot_type == "Filled"

                # Add shaded checkbox that's only enabled when Line is selected
                shaded_disabled = plot_type == "Filled"
                shaded_help = "" if not shaded_disabled else "Enable Line plot type to use shading"
                st.session_state.plot_settings["shaded"] = st.checkbox(
                    "Shaded",
                    value=st.session_state.plot_settings.get("shaded", False),
                    disabled=shaded_disabled,
                    help=shaded_help,
                )

                # Add Axis legend checkbox that's only visible for single 2D plots
                if len(st.session_state.x_params) == 1 and len(st.session_state.y_params) == 1:
                    st.session_state.plot_settings["axis_legend"] = st.checkbox(
                        "Axis legend",
                        value=st.session_state.plot_settings.get("axis_legend", False),
                        help="Place legend inside the plot axes",
                    )

                # Add Z-axis radio and dropdown to match original
                use_z_axis = st.radio(
                    "Use Z-axis",
                    options=["No", "Yes"],
                    index=0 if not st.session_state.plot_settings.get("use_z_axis", False) else 1,
                    horizontal=True,
                )
                st.session_state.plot_settings["use_z_axis"] = use_z_axis == "Yes"

                # Only show Z parameter selection if Z-axis is enabled
                if st.session_state.plot_settings["use_z_axis"]:
                    # Get all available parameters for Z-axis
                    z_params = []
                    if st.session_state.param_names and hasattr(st.session_state.param_names, "list"):
                        z_params.extend(st.session_state.param_names.list())

                    # Add Z parameter dropdown
                    if z_params:
                        z_param = st.session_state.plot_settings.get("z_param", z_params[0] if z_params else None)
                        st.session_state.plot_settings["z_param"] = st.selectbox(
                            "Z-axis parameter",
                            options=z_params,
                            index=z_params.index(z_param) if z_param in z_params else 0,
                        )

                        # Add shadows checkbox for Z-axis plots
                        st.session_state.plot_settings["shadows"] = st.checkbox(
                            "Shadows", value=st.session_state.plot_settings.get("shadows", False)
                        )

                # Number of contours is handled internally by getdist

                # Add Color by radio and dropdown to match original layout
                use_color_by = st.radio(
                    "Color by",
                    options=["No", "Yes"],
                    index=0 if st.session_state.plot_settings.get("color_by", "None") == "None" else 1,
                    horizontal=True,
                )

                # Only show color parameter selection if Color by is enabled
                if use_color_by == "Yes":
                    # Add color by dropdown
                    st.session_state.plot_settings["color_by"] = st.selectbox(
                        "Parameter",
                        options=color_params,
                        index=color_params.index(st.session_state.plot_settings.get("color_by", color_params[0]))
                        if st.session_state.plot_settings.get("color_by", color_params[0]) in color_params
                        else 0,
                    )
                else:
                    st.session_state.plot_settings["color_by"] = "None"

            elif st.session_state.plot_type == "Triangle":
                st.session_state.plot_settings["filled"] = st.checkbox(
                    "Filled Contours", value=st.session_state.plot_settings.get("filled", True)
                )

                st.session_state.plot_settings["show_1d"] = st.checkbox(
                    "Show 1D Distributions", value=st.session_state.plot_settings.get("show_1d", True)
                )

                # Add shaded option for triangle plots
                st.session_state.plot_settings["shaded"] = st.checkbox(
                    "Shaded", value=st.session_state.plot_settings.get("shaded", False)
                )

                # Add Color by radio and dropdown to match original layout
                use_color_by = st.radio(
                    "Color by",
                    options=["No", "Yes"],
                    index=0 if st.session_state.plot_settings.get("color_by", "None") == "None" else 1,
                    horizontal=True,
                )

                # Only show color parameter selection if Color by is enabled
                if use_color_by == "Yes":
                    # Add color by dropdown
                    st.session_state.plot_settings["color_by"] = st.selectbox(
                        "Parameter",
                        options=color_params,
                        index=color_params.index(st.session_state.plot_settings.get("color_by", color_params[0]))
                        if st.session_state.plot_settings.get("color_by", color_params[0]) in color_params
                        else 0,
                    )
                else:
                    st.session_state.plot_settings["color_by"] = "None"

            # Apply settings button
            if st.button("Apply Settings and Generate Plot", key="apply_settings_button", use_container_width=True):
                with st.spinner("Generating plot with new settings..."):
                    # Apply ignore_rows setting to analysis settings if it exists
                    if "ignore_rows" in st.session_state.plot_settings:
                        apply_analysis_settings({"ignore_rows": st.session_state.plot_settings["ignore_rows"]})

                    image_bytes, script = generate_plot()
                    if image_bytes:
                        st.session_state.current_plot = image_bytes
                    if script:
                        st.session_state.current_script = script
                    st.success("Plot updated with new settings!")

        # Plot tab
        with tab1:
            # Check if we need to force a replot due to settings change
            if st.session_state.force_replot and st.session_state.plotter:
                # Reset the force_replot flag
                st.session_state.force_replot = False

                # Store the current plot parameters
                current_params = {}
                if hasattr(st.session_state, "x_params"):
                    current_params["x_params"] = st.session_state.x_params
                if hasattr(st.session_state, "y_params"):
                    current_params["y_params"] = st.session_state.y_params
                if hasattr(st.session_state, "plot_type"):
                    current_params["plot_type"] = st.session_state.plot_type

                # Regenerate the plot with the same parameters
                if current_params:
                    # Use the existing generate_plot function
                    image_bytes, script = generate_plot()

                    # Update the current plot and script
                    if image_bytes:
                        st.session_state.current_plot = image_bytes
                        if script:
                            st.session_state.current_script = script

                    st.success("Plot updated with new settings!")

            if st.session_state.current_plot:
                st.image(st.session_state.current_plot)
            else:
                st.info("Generate a plot using the controls in the sidebar")

        # Script tab
        with tab3:
            if st.session_state.current_script:
                st.code(st.session_state.current_script, language="python")

                # Download script button
                if st.download_button(
                    label="Download Script",
                    data=st.session_state.current_script,
                    file_name="getdist_plot.py",
                    mime="text/plain",
                ):
                    pass
            else:
                st.info("Generate a plot to see the corresponding script")
    else:
        st.info("Select a chain directory to get started")


if __name__ == "__main__":
    main()
