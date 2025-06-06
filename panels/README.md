### AudioCompare

The `AudioCompare` panel is used to examine audio waveforms and spectrograms
in a single experiment or across experiments. See also the built-in Audio Panel.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/audio-compare.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>

<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/built-in-audio-panel.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>

</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/AudioCompare/README.md">README.md</a>
### CompareMaxAccuracyOverTime

The `CompareMaxAccuracyOverTime` panel is used to help track how the
retraining of a model each week compares to the previous week. This panel
creates a scatter plot of the max average of a metric (of your choosing)
over time.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/CompareMaxAccuracyOverTime/compare-max-accuracy-over-time.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/CompareMaxAccuracyOverTime/README.md">README.md</a>
### DataGridViewer

The `DataGridViewer` panel is used to visualize Comet `DataGrids` which
can contain Images, text, and numeric data.

The UX is a sophisticated approach to grouping data to see (and select)
images and other data in a tabular format, with a search feature that
allows fast querying of the data (including metadata) using Python syntax.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/tabular-view.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/group-by.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/image-dialog.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/DataGridViewer/README.md">README.md</a>
### NotebookViewer

The `NotebookViewer` panel is used to render logged Notebooks, either from
[colab.research.google.com](https://colab.research.google.com/) or
any [Jupyter Notebook](https://jupyter.org/).

Comet will automatically log your Colab notebooks, both as a full
history of commenads as `Code.ipynb', but also as a completed notebook
with images and output. For Jupyter, you can use our
[cometx config --auto-log-notebook yes](https://github.com/comet-ml/cometx/blob/main/README.md#cometx-config)


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/NotebookViewer/notebookviewer.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/NotebookViewer/README.md">README.md</a>
### OptimizerAnalysis

The `OptimizerAnalysis` panel is used to explore results from an
Optimizer Search or Sweep. The [Comet Optimizer]() is used to
dynamically find the best set of hyperparameter values that will
minimize a Hyper Parameter Optimization tool (HPO) that can be used to
maximize a particular metric. The OptimizerAnalysis panel, combined
with the [Parallel Coordinate Chart](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/visualizations/parallel-coordinate-chart/)
allows detailed exploration of the results from your grid search or
sweep.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/OptimizerAnalysis/optimizer-analysis.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/OptimizerAnalysis/README.md">README.md</a>
### SaveModelAsArtifact

This panel allows you to save a model as an artifact. Adding
metadata to the model when you log it allows examination,
and saving, by epoch. You can either create a new Artifact,
or use an existing artifact name.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/SaveModelAsArtifact/save-model-as-artifact.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>



For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/SaveModelAsArtifact/README.md">README.md</a>
### TensorboardGroupViewer

The `TensorboardGroupViewer` panel is used to visualize
Tensorboard-logged items inside a Comet Custom Panel, by grouping. This
panel specifically is used to see a group of experiments' log folders.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardGroupViewer/tensorboard-group-viewer.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

First, run your experiment, including writing and logging the
Tensorboard log folder:

```python
# Set up your experiment
writer = tf.summary.create_file_writer("./logs/%s" % experiment.name)
# Log items, including profile, to writer
# Then, log the folder:
experiment.log_tensorflow_folder("./logs")
```

Next, in the Comet UI you use the the "Group experiments" option on
the left-hand side of the project view. Select the group you'd like to
see the profiles. Finally click on "Copy Selected Experiment Logs to
Tensorboard Server" in this panel.


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TensorboardGroupViewer/README.md">README.md</a>
### TensorboardProfileViewer

The `TensorboardProfileViewer` panel is used to visualize Tensorboard
Profile data logged data inside a Comet Custom Panel.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TensorboardProfileViewer/tensorboard-profile-viewer.png"
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

First, run your experiment, including writing and logging the
Tensorboard logdir:

```python
# Set up your experiment and callbacks:
tboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logs,
    histogram_freq=1,
    profile_batch='500,520'
)
model.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
    callbacks = [tboard_callback]
)
# Then, log the folder:
experiment.log_tensorflow_folder("./logs")
```

Finally click on "Select Experiment with log:" in this panel.


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TensorboardProfileViewer/README.md">README.md</a>
### TotalFidelityMetricPlot

The `TotalFidelityMetricPlot` panel is used to plot Total Fidelity Metrics --- metrics that are not sampled in any way.

You can have your Comet Adminstrator turn on "Store metrics without sampling" in the `Admin Dashboard` => `Organization settings`.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/totalfidelity.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/organization-settings.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TotalFidelityMetricPlot/README.md">README.md</a>
