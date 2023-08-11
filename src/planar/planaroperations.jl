# planar versions of tensor operations add!, trace! and contract!
function planaradd!(C::AbstractTensorMap{S,N₁,N₂},
                    A::AbstractTensorMap{S},
                    p::Index2Tuple{N₁,N₂},
                    α,
                    β,
                    backend::Backend...) where {S,N₁,N₂}
    return add_transpose!(C, A, p, α, β, backend...)
end

function planartrace!(C::AbstractTensorMap{S,N₁,N₂}, 
                      A::AbstractTensorMap{S},
                      p::Index2Tuple{N₁,N₂},
                      q::Index2Tuple{N₃,N₃},
                      α,
                      β,
                      backend::Backend...) where {S,N₁,N₂,N₃}
    if BraidingStyle(sectortype(S)) == Bosonic()
        return trace_permute!(C, A, p, q, α, β, backend...)
    end

    @boundscheck begin
        all(i -> space(A, p[1][i]) == space(C, i), 1:N₁) ||
            throw(SpaceMismatch("trace: A = $(codomain(A))←$(domain(A)),
                    C = $(codomain(C))←$(domain(C)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(A, p[2][i]) == space(C, N₁ + i), 1:N₂) ||
            throw(SpaceMismatch("trace: A = $(codomain(A))←$(domain(A)),
                    C = $(codomain(C))←$(domain(C)), p1 = $(p1), p2 = $(p2)"))
        all(i -> space(A, q[1][i]) == dual(space(A, q[2][i])), 1:N₃) ||
            throw(SpaceMismatch("trace: A = $(codomain(A))←$(domain(A)),
                    q1 = $(q1), q2 = $(q2)"))
    end

    if iszero(β)
        fill!(C, β)
    elseif !isone(β)
        rmul!(C, β)
    end
    for (f₁, f₂) in fusiontrees(A)
        for ((f₁′, f₂′), coeff) in planar_trace(f₁, f₂, p..., q...)
            TO.tensortrace!(C[f₁′, f₂′], p, A[f₁, f₂], q, :N, α * coeff, true, backend...)
        end
    end
    return C
end


function planarcontract!(C::AbstractTensorMap{S,N₁,N₂}, 
                         A::AbstractTensorMap{S},
                         pA::Index2Tuple,
                         B::AbstractTensorMap{S},
                         pB::Index2Tuple,
                         pAB::Index2Tuple{N₁,N₂},
                         α,
                         β,
                         backend::Backend...) where {S,N₁,N₂}
    codA, domA = codomainind(A), domainind(A)
    codB, domB = codomainind(B), domainind(B)
    oindA, cindA = pA
    cindB, oindB = pB
    oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, pAB...)

    if oindA == codA && cindA == domA
        A′ = A
    else
        A′ = TO.tensoralloc_add(scalartype(A), (oindA, cindA), A, :N, true)
        add_transpose!(A′, A, (oindA, cindA), true, false, backend...)
    end

    if cindB == codB && oindB == domB
        B′ = B
    else
        B′ = TensorOperations.tensoralloc_add(scalartype(B), (cindB, oindB), B, :N, true)
        add_transpose!(B′, B, (cindB, oindB), true, false, backend...)
    end
    mul!(C, A′, B′, α, β)
    (oindA == codA && cindA == domA) || TO.tensorfree!(A′)
    (cindB == codB && oindB == domB) || TO.tensorfree!(B′)

    return C
end

# function planaradd!(α, tsrc::AbstractTensorMap{S},
#                     β, tdst::AbstractTensorMap{S,N₁,N₂},
#                     p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {S,N₁,N₂}
#     return add_transpose!(α, tsrc, β, tdst, p1, p2)
# end

# function planar_trace!(α, tsrc::AbstractTensorMap{S},
#                        β, tdst::AbstractTensorMap{S,N₁,N₂},
#                        p1::IndexTuple{N₁}, p2::IndexTuple{N₂},
#                        q1::IndexTuple{N₃}, q2::IndexTuple{N₃}) where {S,N₁,N₂,N₃}
#     if BraidingStyle(sectortype(S)) == Bosonic()
#         return trace!(α, tsrc, β, tdst, p1, p2, q1, q2)
#     end

#     @boundscheck begin
#         all(i -> space(tsrc, p1[i]) == space(tdst, i), 1:N₁) ||
#             throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
#                     tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
#         all(i -> space(tsrc, p2[i]) == space(tdst, N₁ + i), 1:N₂) ||
#             throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
#                     tdst = $(codomain(tdst))←$(domain(tdst)), p1 = $(p1), p2 = $(p2)"))
#         all(i -> space(tsrc, q1[i]) == dual(space(tsrc, q2[i])), 1:N₃) ||
#             throw(SpaceMismatch("trace: tsrc = $(codomain(tsrc))←$(domain(tsrc)),
#                     q1 = $(q1), q2 = $(q2)"))
#     end

#     if iszero(β)
#         fill!(tdst, β)
#     elseif β != 1
#         rmul!(tdst, β)
#     end
#     pdata = (p1..., p2...)
#     for (f₁, f₂) in fusiontrees(tsrc)
#         for ((f₁′, f₂′), coeff) in planar_trace(f₁, f₂, p1, p2, q1, q2)
#             TO._trace!(α * coeff, tsrc[f₁, f₂], true, tdst[f₁′, f₂′], pdata, q1, q2)
#         end
#     end
#     return tdst
# end

# function planar_contract!(α, A::AbstractTensorMap{S}, B::AbstractTensorMap{S},
#                           β, C::AbstractTensorMap{S},
#                           oindA::IndexTuple, cindA::IndexTuple,
#                           oindB::IndexTuple, cindB::IndexTuple,
#                           p1::IndexTuple, p2::IndexTuple,
#                           syms::Union{Nothing,NTuple{3,Symbol}}=nothing) where {S}
#     codA, domA = codomainind(A), domainind(A)
#     codB, domB = codomainind(B), domainind(B)
#     oindA, cindA, oindB, cindB = reorder_indices(codA, domA, codB, domB, oindA, cindA,
#                                                  oindB, cindB, p1, p2)

#     if oindA == codA && cindA == domA
#         A′ = A
#     else
#         if isnothing(syms)
#             A′ = TO.similar_from_indices(eltype(A), oindA, cindA, A, :N)
#         else
#             A′ = TO.cached_similar_from_indices(syms[1], eltype(A), oindA, cindA, A, :N)
#         end
#         add_transpose!(true, A, false, A′, oindA, cindA)
#     end
#     if cindB == codB && oindB == domB
#         B′ = B
#     else
#         if isnothing(syms)
#             B′ = TO.similar_from_indices(eltype(B), cindB, oindB, B, :N)
#         else
#             B′ = TO.cached_similar_from_indices(syms[2], eltype(B), cindB, oindB, B, :N)
#         end
#         add_transpose!(true, B, false, B′, cindB, oindB)
#     end
#     mul!(C, A′, B′, α, β)
#     return C
# end

# auxiliary routines
_cyclicpermute(t::Tuple) = (Base.tail(t)..., t[1])
_cyclicpermute(t::Tuple{}) = ()

function reorder_indices(codA, domA, codB, domB, oindA, oindB, p1, p2)
    N₁ = length(oindA)
    N₂ = length(oindB)
    @assert length(p1) == N₁ && all(in(p1), 1:N₁)
    @assert length(p2) == N₂ && all(in(p2), N₁ .+ (1:N₂))
    oindA2 = TupleTools.getindices(oindA, p1)
    oindB2 = TupleTools.getindices(oindB, p2 .- N₁)
    indA = (codA..., reverse(domA)...)
    indB = (codB..., reverse(domB)...)
    # cycle indA to be of the form (oindA2..., reverse(cindA2)...)
    while length(oindA2) > 0 && indA[1] != oindA2[1]
        indA = _cyclicpermute(indA)
    end
    # cycle indB to be of the form (cindB2..., reverse(oindB2)...)
    while length(oindB2) > 0 && indB[end] != oindB2[1]
        indB = _cyclicpermute(indB)
    end
    for i in 2:N₁
        @assert indA[i] == oindA2[i]
    end
    for j in 2:N₂
        @assert indB[end + 1 - j] == oindB2[j]
    end
    Nc = length(indA) - N₁
    @assert Nc == length(indB) - N₂
    pc = ntuple(identity, Nc)
    cindA2 = reverse(TupleTools.getindices(indA, N₁ .+ pc))
    cindB2 = TupleTools.getindices(indB, pc)
    return oindA2, cindA2, oindB2, cindB2
end

function reorder_indices(codA, domA, codB, domB, oindA, cindA, oindB, cindB, p1, p2)
    oindA2, cindA2, oindB2, cindB2 = reorder_indices(codA, domA, codB, domB, oindA, oindB,
                                                     p1, p2)

    #if oindA or oindB are empty, then reorder indices can only order it correctly up to a cyclic permutation!
    if isempty(oindA2) && !isempty(cindA)
        # isempty(cindA) is a cornercase which I'm not sure if we can encounter
        hit = cindA[findfirst(==(first(cindB2)), cindB)]
        while hit != first(cindA2)
            cindA2 = _cyclicpermute(cindA2)
        end
    end
    if isempty(oindB2) && !isempty(cindB)
        hit = cindB[findfirst(==(first(cindA2)), cindA)]
        while hit != first(cindB2)
            cindB2 = _cyclicpermute(cindB2)
        end
    end
    @assert TupleTools.sort(cindA) == TupleTools.sort(cindA2)
    @assert TupleTools.sort(tuple.(cindA2, cindB2)) == TupleTools.sort(tuple.(cindA, cindB))
    return oindA2, cindA2, oindB2, cindB2
end
