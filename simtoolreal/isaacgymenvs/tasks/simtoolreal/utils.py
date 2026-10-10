# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations




def populate_dof_properties(hand_arm_dof_props, arm_dofs: int, hand_dofs: int) -> None:
    assert len(hand_arm_dof_props["stiffness"]) == arm_dofs + hand_dofs

    import numpy as np

    kuka_efforts = [300, 300, 300, 300, 300, 300, 300]
    kuka_stiffnesses = [600, 600, 500, 400, 200, 200, 200]
    kuka_dampings = [
        27.027026473513512,
        27.027026473513512,
        24.672186769721083,
        22.067474708266914,
        9.752538131173853,
        9.147747263670984,
        9.147747263670984,
    ]
    kuka_gear_ratios = [160, 160, 160, 160, 100, 160, 160]
    kuka_rotor_inertias = [
        0.0001321,
        0.0001321,
        0.0001321,
        0.0001321,
        0.0001321,
        0.0000454,
        0.0000454,
    ]

    assert (
        len(kuka_stiffnesses)
        == len(kuka_dampings)
        == len(kuka_gear_ratios)
        == len(kuka_rotor_inertias)
        == arm_dofs
    ), (
        f"{len(kuka_stiffnesses)} != {len(kuka_dampings)} != {len(kuka_gear_ratios)} != {len(kuka_rotor_inertias)} != {arm_dofs}"
    )
    kuka_reflected_inertias = [
        n * n * J for n, J in zip(kuka_gear_ratios, kuka_rotor_inertias)
    ]
    computed_kuka_armatures = kuka_reflected_inertias
    kuka_armatures = [
        3.3817600000000003,
        3.3817600000000003,
        3.3817600000000003,
        3.3817600000000003,
        1.3210000000000002,
        1.16224,
        1.16224,
    ]
    assert np.allclose(computed_kuka_armatures, kuka_armatures), (
        f"computed_kuka_armatures: {computed_kuka_armatures}, kuka_armatures: {kuka_armatures}"
    )

    kuka_damping_ratio = 0.3
    computed_kuka_dampings = [
        2 * kuka_damping_ratio * np.sqrt(kuka_stiffnesses[i] * kuka_armatures[i])
        for i in range(arm_dofs)
    ]
    assert np.allclose(computed_kuka_dampings, kuka_dampings), (
        f"computed_kuka_dampings: {computed_kuka_dampings}, kuka_dampings: {kuka_dampings}"
    )

    hand_arm_dof_props["stiffness"][0:arm_dofs] = kuka_stiffnesses
    hand_arm_dof_props["damping"][0:arm_dofs] = kuka_dampings
    # Not setting armature matches real KUKA robot behavior
    # hand_arm_dof_props["armature"][0:arm_dofs] = kuka_armatures
    hand_arm_dof_props["effort"][0:arm_dofs] = kuka_efforts

    # Assumes hand order
    # ['left_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
    #  'left_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
    #  'left_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
    #  'left_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
    #  'left_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP']
    hand_stiffnesses = [
        6.95,
        13.2,
        4.76,
        6.62,
        0.9,
        4.76,
        6.62,
        0.9,
        0.9,
        4.76,
        6.62,
        0.9,
        0.9,
        4.76,
        6.62,
        0.9,
        0.9,
        1.38,
        4.76,
        6.62,
        0.9,
        0.9,
    ]
    hand_dampings = [
        0.28676845,
        0.40845109,
        0.20394083,
        0.24044435,
        0.04190723,
        0.20859232,
        0.24595532,
        0.04243185,
        0.03504461,
        0.2085923,
        0.24595532,
        0.04243185,
        0.03504461,
        0.20859226,
        0.24595528,
        0.04243183,
        0.0350446,
        0.02782345,
        0.20859229,
        0.24595528,
        0.04243183,
        0.0350446,
    ]
    hand_armatures = [
        0.0032,
        0.0032,
        0.00265,
        0.00265,
        0.0006,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
        0.00012,
        0.00265,
        0.00265,
        0.0006,
        0.00042,
    ]
    hand_frictions = [
        0.132,
        0.132,
        0.07456,
        0.07456,
        0.01276,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
        0.012,
        0.07456,
        0.07456,
        0.01276,
        0.00378738,
    ]
    assert (
        len(hand_stiffnesses)
        == len(hand_dampings)
        == len(hand_armatures)
        == len(hand_frictions)
        == hand_dofs
    ), (
        f"{len(hand_stiffnesses)} != {len(hand_dampings)} != {len(hand_armatures)} != {len(hand_frictions)} != {hand_dofs}"
    )
    hand_arm_dof_props["stiffness"][arm_dofs:] = hand_stiffnesses
    hand_arm_dof_props["damping"][arm_dofs:] = hand_dampings
    hand_arm_dof_props["armature"][arm_dofs:] = hand_armatures
    hand_arm_dof_props["friction"][arm_dofs:] = hand_frictions
