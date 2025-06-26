=====
Usage
=====

To use Conditional KDE in a project::

    from conditional_kde import (
        ConditionalGaussianKernelDensity,
        InterpolatedConditionalKernelDensity,
    )

    kde = ConditionalGaussianKernelDensity()
    kde.fit(data_xyz, features = ["x", "y", "z"])

    kde_intp = InterpolatedConditionalKernelDensity()
    kde_intp.fit(
        data,
        inherent_features = ["z"],
        features = ["x", "y"],
        interpolation_points = {"z": z},
        interpolation_method = "linear",
    )

From there one can either sample the distributions or calculate probability values.
See tutorials for more information.
