using DelimitedFiles
using Plots
using StatsPlots

data =readdlm("perform_table.csv", ',')
display(data[1,:])
data = data[2:end, :]

nodes = log.(2, data[:,1])

plot(title = "CG runtimes by number of elements in each direction", xlabel = "log2 # elements", ylabel = "log2 runtime(s)", legend=:topleft)

plot!(nodes, log.(2, data[:,4]), label="CPU SpMV", shape=:square)
plot!(nodes[1:end-2], log.(2, data[1:end-2,2]), label="GPU SpMV", shape=:circle)
plot!(nodes[1:end-5], log.(2, data[1:end-5,5]), label="CPU Mat-Free", shape=:square)
plot!(nodes[1:end-2], log.(2, data[1:end-2,3]), label="GPU Mat-Free", shape=:circle)
plot!(nodes[1:end-2], log.(2, data[1:end-2,7]), label="Serial SpMV")
plot!(nodes[1:end-5], log.(2, data[1:end-5,8]), label="Serial Mat-Free", color=:black)
display(data)

#gui()



data = data[1:end, 1:end .!=6]
toplot = log.(2, data[5, 2:end])
toplot2 = [i for i in log.(2,data[5,2:end])]

display(toplot)
ctg = repeat(["SpMV", "Mat-Free"], inner = 3)
name = repeat(["CPU", "GPU", "Serial"], outer = 2)
StatsPlots.groupedbar(name, toplot2, group = ctg, title="runtimes of CG on 128 elements per direction", ylabel = "log2 runtime(s)")


png("runtime128")

