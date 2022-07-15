using DelimitedFiles

errors = readdlm("converge_cpu_spmv")
println("spmv convergence")
errors = errors[2:end, :]

for i in 1:size(errors)[1]
   
    println(sqrt(errors[i,2]))
    #=
    if i > 1
        println(errors[i]sqrt(log(2, errors[i-1,2]/errors[i,2])))
    end
    =#
end

errors = readdlm("converge_cpu_matfree")
println("matfree convergence")
        
errors = errors[2:end, :]

for i in 1:size(errors)[1]
    
    if i > 1
        println(log(2, errors[i-1,2]/errors[i,2]))
    end
end
