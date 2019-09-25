function accuracy(y::AbstractVector{T}, ŷ::AbstractVector{T}) where T
    @assert length(y) == length(ŷ)
    return sum(y .== ŷ) / length(y)
end


# true labels on rows, predicted labels on columns
function confusion_matrix(y::AbstractVector{T}, ŷ::AbstractVector{T}; norm=false) where T
    labels = sort(collect(Set(y)))
    index = Dict(l => i for (i, l) in enumerate(labels))
    num_classes = length(labels)
    mat = zeros(Int, num_classes, num_classes)    
    for (t, p) in zip(y, ŷ)
        mat[index[t], index[p]] += 1
    end
    if norm
        mat = mat ./ sum(mat, dims=2)
    end
    return mat
end


function recall(y::AbstractVector{T}, ŷ::AbstractVector{T}) where T
    tn, fn, fp, tp = confusion_matrix(y, ŷ)
    return tp / (tp + fn)
end


function precision(y::AbstractVector{T}, ŷ::AbstractVector{T}) where T
    tn, fn, fp, tp = confusion_matrix(y, ŷ)
    return tp / (tp + fp)
end
