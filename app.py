from flask import Flask, render_template, request, send_from_directory, make_response, redirect, url_for
from werkzeug.utils import secure_filename
import os
import subprocess
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import random
from threading import Thread
from uuid import uuid4
import re
import json
import zipfile

# Configuration and global variables
UPLOAD_FOLDER = '/project/alishredder_data/data_files'
ALISHREDDER_PATH = '/cibiv/www/vhosts/alishredder/AliShredder'
ALLOWED_EXTENSIONS = {'fa', 'faa', 'fas', 'fast', 'fasta', 'fna', 'ph', 'phy', 'phylip'}
SEQUENCE_THRESHOLD = 200

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to count sequences in a FASTA file
def count_sequences_in_fasta(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                count += 1
    return count

# Function to count sequences in a PHYLIP file
def count_sequences_in_phylip(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        count = int(first_line.split()[0])
    return count


# Validate uploaded file
def validate_file(file):
    if file.filename == '':
        return None, "No file selected. Please upload a file."
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        return file, None
    return None, "Invalid file type. Please upload a file in one of the following formats: " + ", ".join(
        ALLOWED_EXTENSIONS)


# Save uploaded file
def save_file(file):
    unique_timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S') + f"{random.randint(1000, 9999)}"
    unique_id = str(uuid4())
    unique_identifier = unique_timestamp + "_" + unique_id
    unique_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)
    os.makedirs(unique_directory, exist_ok=True)
    unique_filename = secure_filename(file.filename)
    filepath = os.path.join(unique_directory, unique_filename)
    file.save(filepath)

    file_extension = unique_filename.rsplit('.', 1)[1].lower()
    if file_extension in ['fa', 'faa', 'fas', 'fast', 'fasta', 'fna']:
        num_sequences = count_sequences_in_fasta(filepath)
    elif file_extension in ['phy', 'phylip', 'ph']:
        num_sequences = count_sequences_in_phylip(filepath)

    log_status_message(unique_directory, f"Number of sequences: {num_sequences} in file: {unique_filename}")

    if num_sequences > SEQUENCE_THRESHOLD:
        error_message = f"File contains too many sequences: {num_sequences}, which exceeds the threshold of {SEQUENCE_THRESHOLD}."
        log_status_message(unique_directory, error_message)
        # Return False to indicate failure along with the error message
        return False, filepath, unique_identifier, unique_filename, error_message

    log_status_message(unique_directory, f"File {unique_filename} uploaded successfully.")
    # Return True to indicate success, and no error message
    return True, filepath, unique_identifier, unique_filename, None


# Create zip file
def create_zip_file(directory, zip_filename, exclude_files=None):
    if exclude_files is None:
        exclude_files = ['status.txt', 'results.json']
    exclude_files.append(zip_filename)  # Ensure the zip file does not zip itself
    zip_file_path = os.path.join(directory, zip_filename)

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file not in exclude_files and file != zip_filename:  # Double check to not include the zip
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory)
                    zipf.write(file_path, arcname=arcname)
    return zip_file_path


# Logging status updates to status.txt
def log_status_message(unique_directory, message):
    status_file_path = os.path.join(unique_directory, 'status.txt')
    with open(status_file_path, 'a') as status_file:
        status_file.write(f"{datetime.utcnow().isoformat()}: {message}\n")


# Function to add a signature line to the log file
def add_signature_to_log(log_file_path, signature_line):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    with open(log_file_path, 'w') as file:
        signature_added = False
        for line in lines:
            file.write(line)
            if 'developed by Olga Chernomor, Arndt von Haeseler and Bui Quang Minh.' in line and not signature_added:
                file.write(signature_line + '\n')
                signature_added = True


# Clean the log file from absolute paths to hide servers directory structure.
def read_and_clean_log(file_path, clean=True):
    try:
        with open(file_path, 'r') as log_file:
            log_content = log_file.readlines()
            cleaned_log_content = []
            for line in log_content:
                if line.startswith('Command:'):
                    match = re.search(r'/([^/]+\.\w+)\s+-nt', line)
                    if match:
                        filename = match.group(1)  # Extracted filename
                        line = f"Command: AliShredder -s {filename} -nt 4\n"
                else:
                    line = re.sub(r'/project/alishredder_data/data_files[^\s]*/([^/ ]+)', r'\1', line)
                cleaned_log_content.append(line)
            return ''.join(cleaned_log_content) if clean else ''.join(log_content)
    except Exception as e:
        return f"Could not read log file: {e}"


# Start a background task for file processing
def start_background_task(filepath, form_data, unique_identifier, unique_filename):
    thread = Thread(target=process_file_background, args=(filepath, form_data, unique_identifier, unique_filename))
    thread.start()
    log_status_message(os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier),
                       f"Background task for {unique_filename} started.")


# Process file in the background
def process_file_background(filepath, form_data, unique_identifier, unique_filename):
    status_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)
    log_status_message(status_dir, "Starting AliShredder execution.")

    try:
        command = build_alishredder_command(filepath, form_data)
        log_status_message(status_dir, f"Executing command: {command}")

        result, error = execute_alishredder(command)
        log_file_path = os.path.join(status_dir, f"{unique_filename}.log")

        # Add the signature line to the log file
        signature_line = "- AliShredder Web Interface developed by Belkis Gueler and Stefan Kalteis"
        add_signature_to_log(log_file_path, signature_line)

        if error:
            log_status_message(status_dir, "AliShredder execution failed.")
            return

        with open(log_file_path, 'r') as log_file:
            log_content = log_file.readlines()
            last_line = log_content[-1] if log_content else ""
            if re.search(r"Date and Time: \w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4}", last_line):
                log_status_message(status_dir, "AliShredder execution completed successfully.")
            else:
                log_status_message(status_dir,
                                   "AliShredder execution may not have completed successfully. Check log file.")
                return

        # Start processing results after successful completion check
        log_status_message(status_dir, "Starting processing results.")
        process_results(os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier), unique_filename)
        log_status_message(status_dir, "Results successfully processed.")

    except Exception as e:
        log_status_message(status_dir, f"Exception during execution: {str(e)}")


# Execute AliShredder with given options
def execute_alishredder(command):
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}, None
    except subprocess.CalledProcessError as e:
        return None, {"stderr": e.stderr if e.stderr else 'Unknown error during command execution.',
                      "returncode": e.returncode}


# Build the command for executing AliShredder
def build_alishredder_command(filepath, form_data):
    command = f"{ALISHREDDER_PATH} -s {filepath} -nt 4"  # Assume -nt 4 is a default option

    # Process each form field
    for key, value in form_data.items():
        # Check if the key ends with '_input', indicating it's an input field associated with a checkbox
        if key.endswith('_input'):
            param_key = key[:-6]  # Get the checkbox name
            if param_key in form_data and form_data[param_key] == 'on':
                # If checkbox is checked, use the input field's value
                command += f" -{param_key} {value}"
        elif value == 'on' and f"{key}_input" not in form_data:
            # If it's a checkbox without an associated input field, add it directly
            command += f" -{key}"

    return command


# Process data and write results to json file
def process_results(unique_directory, unique_filename):
    status_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_directory)
    created_files = [f for f in os.listdir(unique_directory) if os.path.isfile(os.path.join(unique_directory, f))]
    log_status_message(status_dir, f"created files: {created_files}")

    plots, df_html, log_content = {}, "", None

    output_files_chart_functions = {
        "_info_CW.tsv": "chart_all",
        "_CW.tsv": "chart_cw",
        "_DIJ.tsv": "chart_dij",
        "_CIJ.tsv": "chart_cij",
        "_IIJ.tsv": "chart_iij",
        "_CR.tsv": "chart_cr",
        "_info.tsv": "chart_pi"
    }

    for output_suffix, chart_function_name in output_files_chart_functions.items():
        output_filename = f"{unique_filename}{output_suffix}"
        for created_file in created_files:
            if created_file == output_filename:
                file_path = os.path.join(unique_directory, created_file)
                df = pd.read_csv(file_path, sep='\t')

                if chart_function_name == "chart_all":
                    plot_html = globals()[chart_function_name](df, unique_directory).to_html(full_html=False)
                else:
                    plot_html = globals()[chart_function_name](df).to_html(full_html=False)
                plots[chart_function_name] = plot_html
                log_status_message(status_dir, f"Successfully generated plot for {output_filename}.")

    info_cw_files = [file for file in created_files if file.endswith('_info_CW.tsv')]
    if info_cw_files:
        for output_filename in info_cw_files:
            file_path = os.path.join(unique_directory, output_filename)
            df = pd.read_csv(file_path, sep='\t')
            additional_plot_html = chart_all_iq(df).to_html(full_html=False)
            plots["chart_all_iq"] = additional_plot_html
            log_status_message(unique_directory,
                               f"Successfully generated additional plot for {output_filename} using chart_all_iq.")

    if not plots:
        log_file_path = os.path.join(unique_directory, f"{unique_filename}.log")
        log_content = read_and_clean_log(log_file_path)
        log_status_message(status_dir, "Processed log content successfully.")

    log_status_message(status_dir, "Completed processing results.")

    log_status_message(status_dir, "Starting zip file creation.")
    zip_filename = f"{unique_filename}-all_results.zip"
    try:
        create_zip_file(unique_directory, zip_filename,
                        exclude_files=['status.txt', 'results.json'])
        created_files.append(zip_filename)
        log_status_message(status_dir, f"Zip file {zip_filename} created successfully.")
    except Exception as e:
        log_status_message(status_dir, f"Failed to create zip file due to error: {e}")

    results_dict = {
        'created_files': created_files,
        'plots': plots,
        'df_html': df_html,
        'log_content': log_content
    }

    results_file_path = os.path.join(unique_directory, 'results.json')
    with open(results_file_path, 'w') as results_file:
        json.dump(results_dict, results_file)


def chart_all(df, unique_directory):
    dij_file_name = next((file for file in os.listdir(unique_directory) if '_DIJ.tsv' in file), None)
    dij_df = None
    if dij_file_name:
        dij_file_path = os.path.join(unique_directory, dij_file_name)
        try:
            dij_df = pd.read_csv(dij_file_path, sep='\t')
        except Exception as e:
            print(f"Failed to read DIJ file: {e}")

    fig = go.Figure()
    subset_df = df.iloc[1:].copy()
    if dij_df is not None:
        dist_df = avg_d(dij_df)
        subset_df['Average Distance'] = dist_df['Average Distance']

    subset_df['info_total'] = subset_df['info_full'] + subset_df['info_partly']
    subset_df['uninfo_total'] = subset_df['const_full'] + subset_df['const_gapped_wildcard'] + subset_df['uninfo_var']

    score_columns = ['info_full', 'info_partly', 'const_full', 'const_gapped_wildcard', 'uninfo_var', 'info_total',
                     'uninfo_total']
    if 'Average Distance' in subset_df:
        score_columns.append('Average Distance')

    legend_labels = {
        'info_full': 'Fully Informative',
        'info_partly': 'Partly Informative',
        'info_total': 'Informative Total',
        'const_full': 'Fully Conserved',
        'const_gapped_wildcard': 'Conserved with Gaps/Wildcards',
        'uninfo_var': 'Variable, but uninformative',
        'uninfo_total': 'Uninformative Total',
        'Average Distance': 'Average Distance'
    }

    line_styles = {
        'info_total': 'solid',
        'uninfo_total': 'solid',
        'info_full': 'solid',
        'info_partly': 'solid',
        'const_full': 'solid',
        'const_gapped_wildcard': 'solid',
        'uninfo_var': 'solid',
        'Average Distance': 'dash'
    }

    line_colors = {
        'info_full': '#4F7D00',
        'info_partly': '#65D866',
        'info_total': '#007D15',
        'const_full': '#D15A4E',
        'const_gapped_wildcard': '#DD7900',
        'uninfo_var': '#FF6347',
        'uninfo_total': '#A7183A',
        'Average Distance': 'black'
    }

    if 'Cw' in subset_df.columns:
        score_columns.append('Cw')
        legend_labels['Cw'] = 'Completeness'
        line_styles['Cw'] = 'dot'
        line_colors['Cw'] = '#696969'

    for score_type in score_columns:
        if score_type == 'Cw':
            score = subset_df['Cw'] * 100
        elif score_type == 'Average Distance':
            score = subset_df['Average Distance'] * 100
        else:
            score = (subset_df[score_type] / subset_df['len']) * 100

        score.fillna(0, inplace=True)
        subset_df[score_type] = score.astype(int)

        visible_mode_all = 'legendonly' if score_type in ['Cw', 'Average Distance'] else True
        line_width = 4 if score_type in ['info_total', 'uninfo_total', 'Average Distance', 'Cw'] else 2.5

        fig.add_trace(
            go.Scatter(x=subset_df['midpos'], y=subset_df[score_type], mode='lines', name=legend_labels[score_type],
                       line=dict(dash=line_styles[score_type], width=line_width, color=line_colors[score_type]),
                       visible=visible_mode_all))

    fig.update_layout(title='Informative Sites Plot',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='Proportion of Sites [%]',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      hoverlabel=dict(namelength=-1), xaxis=dict(tickmode='auto'))
    return fig


def chart_all_iq(df):
    fig = go.Figure()
    subset_df = df.iloc[1:].copy()  # Optionally skip the first row
    score_columns = ['iq_info', 'iq_const', 'iq_singleton']
    legend_labels = {
        'iq_info': 'Informative (IQ-Tree)',
        'iq_const': 'Conserved (IQ-Tree)',
        'iq_singleton': 'Singleton (IQ-Tree)',
    }

    line_styles = {
        'iq_info': 'solid',
        'iq_const': 'solid',
        'iq_singleton': 'solid',
    }

    for score_type in score_columns:
        # Calculate scores as percentages
        score = (subset_df[score_type] / subset_df['len']) * 100
        subset_df.loc[:, score_type] = score.astype(int)

        # Add trace for each score type
        fig.add_trace(go.Scatter(x=subset_df['midpos'], y=subset_df[score_type], mode='lines',
                                 name=legend_labels[score_type], line=dict(dash=line_styles[score_type])))

    # Layout adjustments for the plot
    fig.update_layout(title='Informative Sites Plot (IQ Values)',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='Completeness Scores',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      hoverlabel=dict(namelength=-1), xaxis=dict(tickmode='auto'))
    return fig


# Completeness Score for the window
def chart_cw(df):
    fig = go.Figure()
    df = df.iloc[1:]
    fig.add_trace(go.Scatter(x=df['midpos'], y=df['Cw'], mode='lines', name='Cw', hoverlabel=dict(namelength=0)))
    fig.update_layout(title='Completeness score for the alignment/window',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='CW',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      hoverlabel=dict(namelength=-1),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      xaxis=dict(tickmode='auto'))
    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(range=[0, 1])
    return fig


# Completeness of Sequence Pairs
def chart_cij(df):
    avg_score_df = pd.DataFrame()
    for i in range(1, df.shape[1] + 1):
        sequence_columns = [col for col in df.columns if col.startswith(f'c{i}_') or col.endswith(f'_{i}')]
        if sequence_columns:
            df[sequence_columns] = df[sequence_columns].apply(pd.to_numeric, errors='coerce')
            avg_score = (df[sequence_columns].sum(axis=1) / (len(sequence_columns)))
            avg_score_df[f'Sequence {i}'] = avg_score
    avg_score_df = pd.concat([df.iloc[3:, :4], avg_score_df], axis=1)

    fig = go.Figure()
    score_columns = avg_score_df.columns[4:]
    for score_type in score_columns:
        fig.add_trace(go.Scatter(x=avg_score_df['midpos'], y=avg_score_df[score_type], mode='lines', name=score_type))

    fig.update_layout(title='Frequency of Sequence Pairs',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='Frequency',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      hoverlabel=dict(namelength=-1),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      xaxis=dict(tickmode='auto'))
    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(range=[0, 1])
    return fig


# Completeness Score for Individual Sequences
def chart_cr(df):
    seq = {}
    for i, col in enumerate(df.columns):
        if col.startswith('s'):
            sequence_number = col[1:]
            new_column_name = 'Sequence ' + sequence_number
            seq[col] = new_column_name
    df.rename(columns=seq, inplace=True)

    fig = go.Figure()
    score_columns = df.columns[4:]

    for score_type in score_columns:
        fig.add_trace(go.Scatter(x=df['midpos'], y=df[score_type], mode='lines', name=score_type))

    fig.update_layout(title='Informative Sites for indiviual Sequences',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='Completeness Scores',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      hoverlabel=dict(namelength=-1),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      xaxis=dict(tickmode='auto'))
    
    fig.update_yaxes(range=[0, 1])

    return fig


# Average Distance
def avg_d(df):
    dist_df = pd.DataFrame()
    df_subset = df.iloc[3:, 4:].astype(float)
    for i in range(1, df_subset.shape[1] + 1):  
        sequence_columns = [col for col in df_subset.columns if col.startswith(f'd{i}_') or col.endswith(f'_{i}')] 
        if sequence_columns:
            # Sum of every row / number of seq pairs 
            length = len(df_subset.columns)
            avg = (df_subset.sum(axis=1))  / length
            dist_df[f'Average Distance'] = avg  
            # Sum of every seq - pair combination / number of seq pairs        
            dist_score = df_subset[sequence_columns].sum(axis=1) / len(sequence_columns)
            dist_df[f'Sequence {i}'] = dist_score               
    result_df = pd.concat([df.iloc[:, :4], dist_df], axis=1)
    result_df = result_df.dropna()       
    return result_df
            


# P-Distance
def chart_dij(df):
    fig = go.Figure()
    df = avg_d(df)
    score_columns = df.columns[4:]
    fig.add_trace(
        go.Scatter(x=df['midpos'], y=df['Average Distance'], mode='lines', name='Average Distance', visible=True,
                   line=dict(color='black', width=4)))
    buttons = []
    buttons.append(dict(label='Show all Sequences',
                        method='update',
                        args=[{'visible': [True] * len(score_columns)},
                              {'title': 'Average P Distance over all Sequence Pairs ',
                               'xaxis': {'title': 'Sequence Position'},
                               'yaxis': {'title': 'P-Distance'}}]))

    for i, score_type in enumerate(score_columns[1:]):
        visible = [False] * len(score_columns)
        visible[i] = True
        buttons.append(dict(label=f'{score_type}',
                            method='update',
                            args=[{'visible': [True] + visible},
                                  {'title': f'Average Distance vs {score_type}',
                                   'xaxis': {'title': 'Sequence Position'},
                                   'yaxis': {'title': 'P-Distance'}}]))

    fig.update_layout(updatemenus=[
        dict(buttons=buttons, direction='down', showactive=True, x=1, xanchor='left', y=1.15, yanchor='top')])

    for score_type in score_columns[1:]:
        visible_mode = 'legendonly'
        fig.add_trace(go.Scatter(x=df['midpos'], y=df[score_type], mode='lines', name=score_type, visible=visible_mode))

    fig.add_trace(
        go.Scatter(x=df['midpos'], y=df['Average Distance'], mode='lines', name='Average Distance', visible=True,
                   line=dict(color='black', width=4), showlegend=False))

    fig.update_layout(title='Average P Distance over all Sequence Pairs ',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='P-Distance',
                      template="plotly_white",
                      legend=dict(traceorder='normal', visible=False),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'downloadPNG', 'downloadSVG', 'downloadExcel', 'zoomIn2d', 'zoomOut2d',
                                   'autoScale2d', 'resetScale2d', 'toggleSpikelines'],
                      hoverlabel=dict(namelength=-1),
                      xaxis=dict(tickmode='auto'))

    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(range=[0, 1])
    return fig


# Incompleteness of Sequence Pairs
def chart_iij(df):
    avg_score_df = pd.DataFrame()
    for i in range(1, df.shape[1] + 1):
        sequence_columns = [col for col in df.columns if col.startswith(f'i{i}_') or col.endswith(f'_{i}')]
        if sequence_columns:
            df[sequence_columns] = df[sequence_columns].apply(pd.to_numeric, errors='coerce')
            avg_score = (df[sequence_columns].sum(axis=1) / (len(sequence_columns)))
            avg_score_df[f'Sequence {i}'] = avg_score

    avg_score_df = pd.concat([df.iloc[3:, :4], avg_score_df], axis=1)
    fig = go.Figure()
    score_columns = avg_score_df.columns[4:]
    for score_type in score_columns:
        fig.add_trace(go.Scatter(x=avg_score_df['midpos'], y=avg_score_df[score_type], mode='lines', name=score_type))

    fig.update_layout(title='Incompleteness for pair of sequences',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='Frequency',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      hoverlabel=dict(namelength=-1), xaxis=dict(tickmode='auto'))
    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(range=[0, 1])
    return fig


# Informative Sites
def chart_pi(df):
    fig = go.Figure()
    subset_df = df.iloc[1:].copy()

    subset_df.loc[:, 'info_total'] = subset_df['info_full'] + subset_df['info_partly']
    subset_df.loc[:, 'uninfo_total'] = subset_df['const_full'] + subset_df['const_gapped_wildcard'] + subset_df[
        'uninfo_var']

    score_columns = ['info_full', 'info_partly', 'const_full', 'const_gapped_wildcard', 'uninfo_var', 'info_total',
                     'uninfo_total']

    legend_labels = {
        'info_full': 'Fully Informative',
        'info_partly': 'Partly Informative',
        'info_total': 'Informative Total',
        'const_full': 'Fully Conserved',
        'const_gapped_wildcard': 'Conserved with Gaps/Wildcards',
        'uninfo_var': 'Variable, but uninformative',
        'uninfo_total': 'Uninformative Total',
    }

    line_styles = {
        'info_total': 'solid',
        'uninfo_total': 'solid',
        'info_full': 'solid',
        'info_partly': 'solid',
        'const_full': 'solid',
        'const_gapped_wildcard': 'solid',
        'uninfo_var': 'solid',
    }

    line_colors = {
        'info_full': '#4F7D00',
        'info_partly': '#65D866',
        'info_total': '#007D15',
        'const_full': '#D15A4E',
        'const_gapped_wildcard': '#DD7900',
        'uninfo_var': '#FF6347',
        'uninfo_total': '#A7183A',
    }

    if 'Cw' in subset_df.columns:
        score_columns.append('Cw')
        legend_labels['Cw'] = 'Completeness'
        line_styles['Cw'] = 'dot'
        line_colors['Cw'] = '#696969'

    for score_type in score_columns:
        if score_type == 'Cw':
            score = subset_df['Cw'] * 100
        else:
            score = (subset_df[score_type] / subset_df['len']) * 100

        score.fillna(0, inplace=True)
        subset_df.loc[:, score_type] = score.astype(int)

        visible_mode_all = 'legendonly' if score_type == 'Cw' else True

        line_width = 4 if score_type in ['info_total', 'uninfo_total', 'Average Distance', 'Cw'] else 2.5

        fig.add_trace(
            go.Scatter(x=subset_df['midpos'], y=subset_df[score_type], mode='lines', name=legend_labels[score_type],
                       line=dict(dash=line_styles[score_type], width=line_width, color=line_colors.get(score_type)),
                       visible=visible_mode_all))

    fig.update_layout(title='Informative Sites Plot',
                      title_x=0.5,
                      xaxis_title='Sequence Position',
                      yaxis_title='Proportion of Sites [%]',
                      template="plotly_white",
                      legend=dict(traceorder='normal'),
                      modebar_add=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape',
                                   'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
                                   'toggleSpikelines'],
                      hoverlabel=dict(namelength=-1), xaxis=dict(tickmode='auto'))

    return fig


# Route for File upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        valid, error = validate_file(file)
        if not valid:
            return render_template('index.html', error=error)

        success, filepath, unique_identifier, unique_filename, error_message = save_file(file)
        if not success:
            # Directly render the fail.html with the error message if not successful
            return render_template('fail.html', log_content=error_message, unique_identifier=unique_identifier)

        # Start a background task to process the file if successful
        start_background_task(filepath, request.form, unique_identifier, unique_filename)
        return redirect(url_for('intermediary_page', unique_identifier=unique_identifier))

    return render_template('index.html')


# Route for fail.html
@app.route('/fail/<unique_identifier>')
def fail(unique_identifier):
    unique_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)

    log_files = [f for f in os.listdir(unique_directory) if f.endswith('.log')]
    if log_files:
        log_file_path = os.path.join(unique_directory, log_files[0])
        log_content = read_and_clean_log(log_file_path)
    else:
        log_content = "Log file not found."

    return render_template('fail.html', error="AliShredder execution failed",
                           log_content=log_content, unique_identifier=unique_identifier)


# Route for intermediary.html
@app.route('/intermediary/<unique_identifier>')
def intermediary_page(unique_identifier):
    unique_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)
    status_file_path = os.path.join(unique_directory, 'status.txt')

    try:
        with open(status_file_path, 'r') as file:
            status_content = file.readlines()  # Read all lines of the file as a list
            exception_message = ""
            execution_failed = False

            for line in status_content:
                if "Exception during execution" in line:
                    exception_message = line.strip()
                    break
                if "AliShredder execution failed." in line:
                    execution_failed = True
                    break

            if exception_message:
                return render_template('fail.html', error="An error occurred during execution.",
                                       log_content=exception_message)
            elif execution_failed:
                return redirect(url_for('fail', unique_identifier=unique_identifier))
            elif any("Results successfully processed." in line for line in status_content):
                return redirect(url_for('success', unique_identifier=unique_identifier))
            else:
                return render_template('intermediary.html', unique_identifier=unique_identifier)
    except FileNotFoundError:
        # Handle the case where the status file is not found
        return "Status file not found", 404


# Route for success.html
@app.route('/success/<unique_identifier>')
def success(unique_identifier):
    unique_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)
    results_file_path = os.path.join(unique_directory, 'results.json')

    try:
        with open(results_file_path, 'r') as results_file:
            results = json.load(results_file)
    except FileNotFoundError as e:
        return render_template('fail.html', log_content="Results have expired. Please run AliShredder again.")
    except json.JSONDecodeError as e:
        return render_template('fail.html', log_content=f"Error reading the results: {e}")

    # Continue as before if no error occurs
    created_files = results.get('created_files', [])
    filtered_files = sorted([file for file in created_files if file not in ['status.txt', 'results.json']])

    zip_files = [file for file in filtered_files if file.endswith('.zip')]
    if zip_files:
        zip_file = zip_files[0]
        filtered_files.remove(zip_file)
        filtered_files.insert(0, zip_file)

    return render_template('success.html',
                           created_files=filtered_files,
                           plots=results.get('plots', {}),
                           df_html=results.get('df_html', ''),
                           log_content=results.get('log_content', ''),
                           unique_identifier=unique_identifier)


# Route for download options.
@app.route('/download/<unique_identifier>/<filename>')
def download_file(unique_identifier, filename):
    unique_directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)
    log_files = [f for f in os.listdir(unique_directory) if f.endswith('.log')]
    log_file_path = os.path.join(unique_directory, log_files[0])

    directory = os.path.join(app.config['UPLOAD_FOLDER'], unique_identifier)
    file_path = os.path.join(directory, filename)

    if filename.endswith('.log'):
        try:
            cleaned_log_content = read_and_clean_log(log_file_path)

            response = make_response(cleaned_log_content)
            response.headers['Content-Type'] = 'text/plain'
            response.headers['Content-Disposition'] = f'attachment; filename={filename}'

            return response
        except IOError:
            return "File not found", 404
    else:
        try:
            return send_from_directory(directory, filename, as_attachment=True)
        except FileNotFoundError:
            return "File not found", 404


# Route for user_manual download
@app.route('/user_manual')
def download_user_manual():
    PDF_DIRECTORY = '/project/alishredder_data'
    return send_from_directory(PDF_DIRECTORY, 'user_manual.pdf')


if __name__ == '__main__':
    app.run(debug=False)
