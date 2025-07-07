import arviz as az
import numpy as np
from arviz.labels import BaseLabeller

from getdist import MCSamples


def _escape_latex_text_for_subscript(text: str) -> str:
    # (Same as before)
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("_", r"\_"),
        ("^", r"\^{}"),
        ("~", r"\textasciitilde{}"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("&", r"\&"),
    ]
    for char, escaped_char in replacements:
        text = text.replace(char, escaped_char)
    return text


def arviz_to_mcsamples(
    idata: az.InferenceData,
    var_names: list[str] | None = None,
    custom_labels: dict[str, str] | None = None,
    custom_ranges: dict[str, tuple[float | None, float | None]] | None = None,
    dataset_label: str | None = None,
    include_coords_in_name: bool = False,
    weights_var: str | None = None,
    loglikes_var: str | None = None,
    arviz_labeller: BaseLabeller | None = None,
) -> MCSamples:
    """
    Convert ArviZ InferenceData to GetDist MCSamples object.

    This function converts MCMC samples stored in ArviZ's InferenceData format
    to GetDist's MCSamples format, enabling the use of GetDist's analysis and
    plotting capabilities with samples from PyMC and other
    samplers that output ArviZ-compatible data.

    :param idata: ArviZ InferenceData object containing posterior samples.
                  Must have a 'posterior' group with 'chain' and 'draw' dimensions.
    :param var_names: List of variable names to include. If None, all variables
                      in the posterior are included.
    :param custom_labels: Dictionary mapping parameter names to custom LaTeX labels
                          for plotting. Keys should match the flattened parameter names.
    :param custom_ranges: Dictionary mapping parameter names to (min, max) tuples
                          for parameter ranges. Either min or max can be None for
                          one-sided bounds.
    :param dataset_label: Label for the dataset, used in plots and output.
                          If None, defaults to "MCSamples from InferenceData".
    :param include_coords_in_name: If True, coordinate values are included in
                                   parameter names for multi-dimensional variables.
                                   If False, uses index-based naming.
    :param weights_var: Name of variable in InferenceData to use as sample weights.
                        If None, all samples are equally weighted.
    :param loglikes_var: Name of variable in InferenceData containing log-likelihood
                         values. If None, no likelihood information is included.
    :param arviz_labeller: Optional ArviZ BaseLabeller instance for parameter
                           labeling. Is used where custom_labels not provided.
    :return: MCSamples object containing the converted samples, ready for GetDist
             analysis and plotting.

    .. note::
        Multi-dimensional parameters are automatically flattened with appropriate
        naming. For example, a parameter 'theta' with shape (3,) becomes
        'theta_0', 'theta_1', 'theta_2'.
    """
    if not hasattr(idata, "posterior"):
        raise ValueError("InferenceData object must have a 'posterior' group.")
    if "chain" not in idata.posterior.sizes or "draw" not in idata.posterior.sizes:
        raise ValueError("InferenceData posterior must have 'chain' and 'draw' dimensions.")

    n_chains = idata.posterior.sizes["chain"]
    n_draws = idata.posterior.sizes["draw"]

    if custom_labels is None:
        custom_labels = {}
    if custom_ranges is None:
        custom_ranges = {}

    final_param_names = []
    final_param_labels_map = {}
    final_param_ranges_map = {}

    if var_names is None:
        target_vars = list(idata.posterior.data_vars)
    else:
        target_vars = var_names

    for var_name in target_vars:
        if var_name not in idata.posterior:
            print(f"Warning: Variable '{var_name}' not found in idata.posterior. Skipping.")
            continue

        data_array = idata.posterior[var_name]
        param_core_dims_shape = data_array.shape[2:]
        n_scalar_params_this_var = int(np.prod(param_core_dims_shape)) if param_core_dims_shape else 1

        if n_scalar_params_this_var == 1 and not param_core_dims_shape:  # True scalar
            generated_name = var_name
            final_param_names.append(generated_name)

            current_default_label = ""
            if arviz_labeller:
                # For scalars, sel and isel are empty
                current_default_label = arviz_labeller.make_label_flat(var_name, {}, {})

            if not current_default_label:
                if var_name.startswith("\\"):
                    current_default_label = var_name
                else:
                    escaped_var_name = var_name.replace("_", r"\_")
                    current_default_label = rf"\text{{{escaped_var_name}}}"

            final_param_labels_map[generated_name] = custom_labels.get(generated_name, current_default_label)
            if generated_name in custom_ranges:
                final_param_ranges_map[generated_name] = custom_ranges[generated_name]

        else:
            core_dim_names = data_array.dims[2:]
            custom_fallback_label_base: str
            if var_name.startswith("\\"):
                custom_fallback_label_base = var_name
            else:
                escaped_var_name_for_custom_base = var_name.replace("_", r"\_")
                custom_fallback_label_base = rf"\text{{{escaped_var_name_for_custom_base}}}"

            for i, multi_idx_tuple in enumerate(np.ndindex(param_core_dims_shape)):
                idx_str_parts_for_name = []
                if include_coords_in_name and core_dim_names and len(core_dim_names) == len(multi_idx_tuple):
                    for dim_idx, coord_dim_name_for_internal_name in enumerate(core_dim_names):
                        if coord_dim_name_for_internal_name in data_array.coords:
                            coord_value_for_name = data_array.coords[coord_dim_name_for_internal_name].values[
                                multi_idx_tuple[dim_idx]
                            ]
                            sanitized_coord_value = (
                                str(coord_value_for_name).replace(" ", "_").replace(".", "p").replace("\\", "")
                            )
                            idx_str_parts_for_name.append(f"{coord_dim_name_for_internal_name}_{sanitized_coord_value}")
                        else:
                            idx_str_parts_for_name.append(f"{core_dim_names[dim_idx]}_{multi_idx_tuple[dim_idx]}")
                else:
                    idx_str_parts_for_name = [str(idx_val) for idx_val in multi_idx_tuple]

                idx_suffix_for_name = "_".join(idx_str_parts_for_name)
                generated_name = f"{var_name}_{idx_suffix_for_name}" if idx_suffix_for_name else var_name
                final_param_names.append(generated_name)

                current_default_label = ""
                if arviz_labeller:
                    selection_for_arviz = {}
                    iselection_for_arviz = {}  # For integer indices

                    # Ensure core_dim_names has the same length as multi_idx_tuple for safe access
                    if len(core_dim_names) == len(multi_idx_tuple):
                        for dim_idx, dim_name in enumerate(core_dim_names):
                            idx_val = multi_idx_tuple[dim_idx]
                            iselection_for_arviz[dim_name] = idx_val  # Populate isel with dim_name: index
                            if dim_name in data_array.coords:
                                selection_for_arviz[dim_name] = data_array.coords[dim_name].values[idx_val]
                            else:
                                selection_for_arviz[dim_name] = idx_val
                    else:
                        # Fallback if core_dim_names doesn't match multi_idx_tuple structure
                        # (e.g. unnamed dimensions beyond chain/draw)
                        selection_for_arviz = {f"dim_{d}": multi_idx_tuple[d] for d in range(len(multi_idx_tuple))}
                        iselection_for_arviz = {f"dim_{d}": multi_idx_tuple[d] for d in range(len(multi_idx_tuple))}

                    current_default_label = arviz_labeller.make_label_flat(
                        var_name, selection_for_arviz, iselection_for_arviz
                    )

                if not current_default_label:
                    latex_subscript_components = []
                    if len(core_dim_names) == len(multi_idx_tuple):
                        for dim_idx, coord_dim_name_for_label in enumerate(core_dim_names):
                            idx_in_this_dim = multi_idx_tuple[dim_idx]
                            if coord_dim_name_for_label in data_array.coords:
                                coord_actual_value = data_array.coords[coord_dim_name_for_label].values[idx_in_this_dim]
                                coord_str_representation = str(coord_actual_value)
                                is_numeric = isinstance(coord_actual_value, (int, float, np.number))

                                if coord_str_representation.startswith("\\"):
                                    latex_subscript_components.append(coord_str_representation)
                                elif is_numeric:
                                    latex_subscript_components.append(coord_str_representation)
                                elif isinstance(coord_actual_value, (str, bytes)):
                                    escaped_text = _escape_latex_text_for_subscript(coord_str_representation)
                                    latex_subscript_components.append(rf"\text{{{escaped_text}}}")
                                else:
                                    escaped_text = _escape_latex_text_for_subscript(coord_str_representation)
                                    latex_subscript_components.append(rf"\text{{{escaped_text}}}")
                            else:
                                latex_subscript_components.append(str(idx_in_this_dim))
                    else:
                        latex_subscript_components = [str(idx_val) for idx_val in multi_idx_tuple]

                    if latex_subscript_components:
                        subscript_string = ",".join(latex_subscript_components)
                        current_default_label = rf"{custom_fallback_label_base}_{{{subscript_string}}}"
                    else:
                        current_default_label = custom_fallback_label_base

                final_param_labels_map[generated_name] = custom_labels.get(generated_name, current_default_label)
                if generated_name in custom_ranges:
                    final_param_ranges_map[generated_name] = custom_ranges[generated_name]

    # ... (rest of the function remains the same: sample prep, weights, loglikes, MCSamples creation)
    if not final_param_names:
        raise ValueError("No parameters selected or found to process after filtering and naming.")

    final_param_labels_list = [final_param_labels_map[name] for name in final_param_names]

    list_of_chain_sample_arrays = []
    for c in range(n_chains):
        current_chain_params_list = []
        for var_name_in_order in target_vars:
            if var_name_in_order not in idata.posterior:
                continue
            data_array_samples = idata.posterior[var_name_in_order]
            samples_this_var_this_chain = data_array_samples.values[c, :, ...]

            param_core_dims_shape_samples = samples_this_var_this_chain.shape[1:]
            n_scalar_params_this_var_samples = (
                int(np.prod(param_core_dims_shape_samples)) if param_core_dims_shape_samples else 1
            )

            reshaped_samples = samples_this_var_this_chain.reshape(n_draws, n_scalar_params_this_var_samples)
            current_chain_params_list.append(reshaped_samples)

        if not current_chain_params_list and final_param_names:
            raise ValueError(f"Internal error: No samples collected for chain {c} despite having parameter names.")

        if current_chain_params_list:
            list_of_chain_sample_arrays.append(np.concatenate(current_chain_params_list, axis=1))
        elif final_param_names:
            pass

    final_samples_arg = list_of_chain_sample_arrays
    if not final_samples_arg and final_param_names:
        if n_chains > 0 and n_draws > 0:
            raise ValueError(
                "Samples array is empty, but parameter names were generated. Check input data and variable processing."
            )

    final_weights_arg = None
    if weights_var:
        weights_da = None
        if "sample_stats" in idata and weights_var in idata.sample_stats:
            weights_da = idata.sample_stats[weights_var]
        elif "posterior" in idata and weights_var in idata.posterior:
            weights_da = idata.posterior[weights_var]
        else:
            raise ValueError(f"Weights variable '{weights_var}' not found in idata.sample_stats or idata.posterior.")

        weights_raw = weights_da.values
        if not (weights_raw.ndim == 2 and weights_raw.shape[0] == n_chains and weights_raw.shape[1] == n_draws):
            raise ValueError(
                f"Weights variable '{weights_var}' must have shape (chain, draw). Found {weights_raw.shape}."
            )
        final_weights_arg = [weights_raw[c, :] for c in range(n_chains)]

    final_loglikes_arg = None
    if loglikes_var:
        log_likelihood_da = None
        if "log_likelihood" in idata and loglikes_var in idata.log_likelihood:
            log_likelihood_da = idata.log_likelihood[loglikes_var]
        elif "sample_stats" in idata and loglikes_var in idata.sample_stats:
            log_likelihood_da = idata.sample_stats[loglikes_var]
        else:
            raise ValueError(
                f"LogLikes variable '{loglikes_var}' not found in idata.log_likelihood or idata.sample_stats."
            )

        log_likelihood_raw = log_likelihood_da.values
        if log_likelihood_raw.ndim > 2:
            log_likelihood_raw = np.sum(log_likelihood_raw, axis=tuple(range(2, log_likelihood_raw.ndim)))

        if not (
            log_likelihood_raw.ndim == 2
            and log_likelihood_raw.shape[0] == n_chains
            and log_likelihood_raw.shape[1] == n_draws
        ):
            raise ValueError(
                f"LogLikes variable '{loglikes_var}' (after sum) must have shape (chain, draw). Found {log_likelihood_raw.shape}."
            )
        neg_loglikes_raw = -log_likelihood_raw
        final_loglikes_arg = [neg_loglikes_raw[c, :] for c in range(n_chains)]

    if n_chains == 1:
        if final_samples_arg:
            final_samples_arg = final_samples_arg[0]
        if final_weights_arg:
            final_weights_arg = final_weights_arg[0]
        if final_loglikes_arg:
            final_loglikes_arg = final_loglikes_arg[0]

    mcs = MCSamples(
        samples=final_samples_arg if final_samples_arg else np.empty((0, len(final_param_names))),
        weights=final_weights_arg,
        loglikes=final_loglikes_arg,
        names=final_param_names,
        labels=final_param_labels_list,
        ranges=final_param_ranges_map if final_param_ranges_map else None,
        label=dataset_label if dataset_label else "MCSamples from InferenceData",
    )
    return mcs
