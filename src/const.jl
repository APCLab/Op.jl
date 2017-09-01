const out_dir = joinpath(Pkg.dir("Op"), "out")
const data_dir = joinpath(Pkg.dir("Op"), "data")
const data_jld = joinpath(data_dir, "data.jld")
const plot_size = (1200, 800)
const plot_rng = 1000  # ploting range for x and y axis
