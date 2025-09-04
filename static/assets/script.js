// Upload - Section
function displayFileName() {
    const fileInput = document.getElementById('file');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    const fileLabel = document.getElementById('fileLabel');
    const fileName = fileInput.files[0].name;
    fileLabel.textContent = fileName; // Update the label text to display the filename
  }

function showPlot(plotName) {
    var plots = ['plot_all', 'plot_dij', 'plot_cij', 'plot_iij', 'plot_cr', 'plot_cw', 'plot_pi', 'plot_all_iq'];
    for (var i = 0; i < plots.length; i++) {
        var plotId = plots[i];
        var plotElement = document.getElementById(plotId);
        if (plotElement) {
            plotElement.style.display = plotId === plotName ? 'block' : 'none';
            if (plotId === plotName) {
                triggerResizeEvent();
            }
        }
    }
}

// JavaScript function to enforce mutual exclusivity between two checkboxes
function handleCheckboxChange(checkedBox, otherBoxId) {
    var otherBox = document.getElementById(otherBoxId);
    if (checkedBox.checked) {
        otherBox.checked = false;
    }
}

function toggleDataFrame() {
    var dataframeContent = document.getElementById("dataframe-content");
    if (dataframeContent) {
        dataframeContent.style.display = dataframeContent.style.display === "none" ? "block" : "none";
    }
}

// New function to automatically display the plot if it's the only one
function displaySinglePlotIfApplicable() {
    var plotSelector = document.getElementById('plotSelector');
    if (plotSelector && plotSelector.options.length === 2) {
        showPlot(plotSelector.options[1].value);
    }
}

function autoCheckCheckbox(inputElement, checkboxId) {
    var checkbox = document.getElementById(checkboxId);
    if (inputElement.value !== "") {
        checkbox.checked = true;
    } else {
        // Optional: Uncheck if the input is cleared
        checkbox.checked = false;
    }
}


function triggerResizeEvent() {
  window.dispatchEvent(new Event('resize'));
}

// Call the new function on page load
document.addEventListener('DOMContentLoaded', function() {
    displaySinglePlotIfApplicable();
});