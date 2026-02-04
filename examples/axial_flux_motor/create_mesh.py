import numpy as np
import gmsh
import sys
import orientation


class AFPM_Mesh_12S5PP:
    """12 Slot 5 PP magnet motor mesh"""

    def __init__(
        self,
        total_length,
        airgap,
        copper_slot_height,
        tooth_tip_thickness,
        bell_width,
        tooth_width,
        magnet_length,
        magnet_thickness,
        back_iron_thickness,
        mesh_refinement,
        npts_airgap,
        gmsh_popup,
    ):
        self.total_length = total_length  # Total slice length
        self.airgap = airgap
        self.copper_slot_height = copper_slot_height
        self.tooth_tip_thickness = tooth_tip_thickness
        self.bell_width = bell_width
        self.tooth_width = tooth_width
        self.magnet_length = magnet_length
        self.magnet_thickness = magnet_thickness
        self.back_iron_thickness = back_iron_thickness
        self.mesh_refinement = mesh_refinement
        self.npts_airgap = npts_airgap
        self.gmsh_popup = gmsh_popup

        # Raise exceptions for invalid geometru
        if magnet_length >= (total_length / 10.0):
            raise Exception("Invalid magnet length")

        if bell_width >= (total_length / 12.0):
            raise Exception("Invalid bell width")

        if tooth_width >= (total_length / 12.0):
            raise Exception("Invalid tooth width")

        return

    def stator(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("stator")
        lc = self.mesh_refinement

        # Dimensions
        L = self.total_length  # Total length of the slice
        Sh = self.copper_slot_height  # Copper slot heigh
        tt = self.tooth_tip_thickness  # Tooth tip thickness
        bw = self.bell_width  # Bell width
        tw = self.tooth_width  # Tooth width
        t_ag = self.airgap  # airgap thickness
        a = 0.5 * (bw - tw)  # Shoe over hang
        c = (L - 12 * bw) / 12  # Slot opening between teeth
        npts_airgap = self.npts_airgap  # Number of points along the airgap mesh

        start = tw * 0.5
        gmsh.model.geo.addPoint(start, Sh * 0.5, 0, lc, 1)
        gmsh.model.geo.addPoint(start + a, Sh * 0.5, 0, lc, 2)
        gmsh.model.geo.addPoint(start + a + c, Sh * 0.5, 0, lc, 3)
        gmsh.model.geo.addPoint(start + 2 * a + c, Sh * 0.5, 0, lc, 4)
        gmsh.model.geo.addPoint(start + 2 * a + c + tw, Sh * 0.5, 0, lc, 5)
        gmsh.model.geo.addPoint(start + 3 * a + c + tw, Sh * 0.5, 0, lc, 6)
        gmsh.model.geo.addPoint(start + 3 * a + 2 * c + tw, Sh * 0.5, 0, lc, 7)
        gmsh.model.geo.addPoint(start + 4 * a + 2 * c + tw, Sh * 0.5, 0, lc, 8)
        gmsh.model.geo.addPoint(start + 4 * a + 2 * c + 2 * tw, Sh * 0.5, 0, lc, 9)
        gmsh.model.geo.addPoint(start + 5 * a + 2 * c + 2 * tw, Sh * 0.5, 0, lc, 10)
        gmsh.model.geo.addPoint(start + 5 * a + 3 * c + 2 * tw, Sh * 0.5, 0, lc, 11)
        gmsh.model.geo.addPoint(start + 6 * a + 3 * c + 2 * tw, Sh * 0.5, 0, lc, 12)
        gmsh.model.geo.addPoint(start + 6 * a + 3 * c + 3 * tw, Sh * 0.5, 0, lc, 13)
        gmsh.model.geo.addPoint(start + 7 * a + 3 * c + 3 * tw, Sh * 0.5, 0, lc, 14)
        gmsh.model.geo.addPoint(start + 7 * a + 4 * c + 3 * tw, Sh * 0.5, 0, lc, 15)
        gmsh.model.geo.addPoint(start + 8 * a + 4 * c + 3 * tw, Sh * 0.5, 0, lc, 16)
        gmsh.model.geo.addPoint(start + 8 * a + 4 * c + 4 * tw, Sh * 0.5, 0, lc, 17)
        gmsh.model.geo.addPoint(start + 9 * a + 4 * c + 4 * tw, Sh * 0.5, 0, lc, 18)
        gmsh.model.geo.addPoint(start + 9 * a + 5 * c + 4 * tw, Sh * 0.5, 0, lc, 19)
        gmsh.model.geo.addPoint(start + 10 * a + 5 * c + 4 * tw, Sh * 0.5, 0, lc, 20)
        gmsh.model.geo.addPoint(start + 10 * a + 5 * c + 5 * tw, Sh * 0.5, 0, lc, 21)
        gmsh.model.geo.addPoint(start + 11 * a + 5 * c + 5 * tw, Sh * 0.5, 0, lc, 22)
        gmsh.model.geo.addPoint(start + 11 * a + 6 * c + 5 * tw, Sh * 0.5, 0, lc, 23)
        gmsh.model.geo.addPoint(start + 12 * a + 6 * c + 5 * tw, Sh * 0.5, 0, lc, 24)
        gmsh.model.geo.addPoint(start + 12 * a + 6 * c + 6 * tw, Sh * 0.5, 0, lc, 25)
        gmsh.model.geo.addPoint(start + 13 * a + 6 * c + 6 * tw, Sh * 0.5, 0, lc, 26)
        gmsh.model.geo.addPoint(start + 13 * a + 7 * c + 6 * tw, Sh * 0.5, 0, lc, 27)
        gmsh.model.geo.addPoint(start + 14 * a + 7 * c + 6 * tw, Sh * 0.5, 0, lc, 28)
        gmsh.model.geo.addPoint(start + 14 * a + 7 * c + 7 * tw, Sh * 0.5, 0, lc, 29)
        gmsh.model.geo.addPoint(start + 15 * a + 7 * c + 7 * tw, Sh * 0.5, 0, lc, 30)
        gmsh.model.geo.addPoint(start + 15 * a + 8 * c + 7 * tw, Sh * 0.5, 0, lc, 31)
        gmsh.model.geo.addPoint(start + 16 * a + 8 * c + 7 * tw, Sh * 0.5, 0, lc, 32)
        gmsh.model.geo.addPoint(start + 16 * a + 8 * c + 8 * tw, Sh * 0.5, 0, lc, 33)
        gmsh.model.geo.addPoint(start + 17 * a + 8 * c + 8 * tw, Sh * 0.5, 0, lc, 34)
        gmsh.model.geo.addPoint(start + 17 * a + 9 * c + 8 * tw, Sh * 0.5, 0, lc, 35)
        gmsh.model.geo.addPoint(start + 18 * a + 9 * c + 8 * tw, Sh * 0.5, 0, lc, 36)
        gmsh.model.geo.addPoint(start + 18 * a + 9 * c + 9 * tw, Sh * 0.5, 0, lc, 37)
        gmsh.model.geo.addPoint(start + 19 * a + 9 * c + 9 * tw, Sh * 0.5, 0, lc, 38)
        gmsh.model.geo.addPoint(start + 19 * a + 10 * c + 9 * tw, Sh * 0.5, 0, lc, 39)
        gmsh.model.geo.addPoint(start + 20 * a + 10 * c + 9 * tw, Sh * 0.5, 0, lc, 40)
        gmsh.model.geo.addPoint(start + 20 * a + 10 * c + 10 * tw, Sh * 0.5, 0, lc, 41)
        gmsh.model.geo.addPoint(start + 21 * a + 10 * c + 10 * tw, Sh * 0.5, 0, lc, 42)
        gmsh.model.geo.addPoint(start + 21 * a + 11 * c + 10 * tw, Sh * 0.5, 0, lc, 43)
        gmsh.model.geo.addPoint(start + 22 * a + 11 * c + 10 * tw, Sh * 0.5, 0, lc, 44)
        gmsh.model.geo.addPoint(start + 22 * a + 11 * c + 11 * tw, Sh * 0.5, 0, lc, 45)
        gmsh.model.geo.addPoint(start + 23 * a + 11 * c + 11 * tw, Sh * 0.5, 0, lc, 46)
        gmsh.model.geo.addPoint(start + 23 * a + 12 * c + 11 * tw, Sh * 0.5, 0, lc, 47)
        gmsh.model.geo.addPoint(start + 24 * a + 12 * c + 11 * tw, Sh * 0.5, 0, lc, 48)

        start2 = bw * 0.5
        gmsh.model.geo.addPoint(0, 0.5 * Sh + tt, 0, lc, 49)
        gmsh.model.geo.addPoint(start2, 0.5 * Sh + tt, 0, lc, 50)
        gmsh.model.geo.addPoint(start2 + c, 0.5 * Sh + tt, 0, lc, 51)
        gmsh.model.geo.addPoint(start2 + c + bw, 0.5 * Sh + tt, 0, lc, 52)
        gmsh.model.geo.addPoint(start2 + 2 * c + bw, 0.5 * Sh + tt, 0, lc, 53)
        gmsh.model.geo.addPoint(start2 + 2 * c + 2 * bw, 0.5 * Sh + tt, 0, lc, 54)
        gmsh.model.geo.addPoint(start2 + 3 * c + 2 * bw, 0.5 * Sh + tt, 0, lc, 55)
        gmsh.model.geo.addPoint(start2 + 3 * c + 3 * bw, 0.5 * Sh + tt, 0, lc, 56)
        gmsh.model.geo.addPoint(start2 + 4 * c + 3 * bw, 0.5 * Sh + tt, 0, lc, 57)
        gmsh.model.geo.addPoint(start2 + 4 * c + 4 * bw, 0.5 * Sh + tt, 0, lc, 58)
        gmsh.model.geo.addPoint(start2 + 5 * c + 4 * bw, 0.5 * Sh + tt, 0, lc, 59)
        gmsh.model.geo.addPoint(start2 + 5 * c + 5 * bw, 0.5 * Sh + tt, 0, lc, 60)
        gmsh.model.geo.addPoint(start2 + 6 * c + 5 * bw, 0.5 * Sh + tt, 0, lc, 61)
        gmsh.model.geo.addPoint(start2 + 6 * c + 6 * bw, 0.5 * Sh + tt, 0, lc, 62)
        gmsh.model.geo.addPoint(start2 + 7 * c + 6 * bw, 0.5 * Sh + tt, 0, lc, 63)
        gmsh.model.geo.addPoint(start2 + 7 * c + 7 * bw, 0.5 * Sh + tt, 0, lc, 64)
        gmsh.model.geo.addPoint(start2 + 8 * c + 7 * bw, 0.5 * Sh + tt, 0, lc, 65)
        gmsh.model.geo.addPoint(start2 + 8 * c + 8 * bw, 0.5 * Sh + tt, 0, lc, 66)
        gmsh.model.geo.addPoint(start2 + 9 * c + 8 * bw, 0.5 * Sh + tt, 0, lc, 67)
        gmsh.model.geo.addPoint(start2 + 9 * c + 9 * bw, 0.5 * Sh + tt, 0, lc, 68)
        gmsh.model.geo.addPoint(start2 + 10 * c + 9 * bw, 0.5 * Sh + tt, 0, lc, 69)
        gmsh.model.geo.addPoint(start2 + 10 * c + 10 * bw, 0.5 * Sh + tt, 0, lc, 70)
        gmsh.model.geo.addPoint(start2 + 11 * c + 10 * bw, 0.5 * Sh + tt, 0, lc, 71)
        gmsh.model.geo.addPoint(start2 + 11 * c + 11 * bw, 0.5 * Sh + tt, 0, lc, 72)
        gmsh.model.geo.addPoint(start2 + 12 * c + 11 * bw, 0.5 * Sh + tt, 0, lc, 73)
        gmsh.model.geo.addPoint(2 * start2 + 12 * c + 11 * bw, 0.5 * Sh + tt, 0, lc, 74)

        gmsh.model.geo.addPoint(start, -Sh * 0.5, 0, lc, 75)
        gmsh.model.geo.addPoint(start + a, -Sh * 0.5, 0, lc, 76)
        gmsh.model.geo.addPoint(start + a + c, -Sh * 0.5, 0, lc, 77)
        gmsh.model.geo.addPoint(start + 2 * a + c, -Sh * 0.5, 0, lc, 78)
        gmsh.model.geo.addPoint(start + 2 * a + c + tw, -Sh * 0.5, 0, lc, 79)
        gmsh.model.geo.addPoint(start + 3 * a + c + tw, -Sh * 0.5, 0, lc, 80)
        gmsh.model.geo.addPoint(start + 3 * a + 2 * c + tw, -Sh * 0.5, 0, lc, 81)
        gmsh.model.geo.addPoint(start + 4 * a + 2 * c + tw, -Sh * 0.5, 0, lc, 82)
        gmsh.model.geo.addPoint(start + 4 * a + 2 * c + 2 * tw, -Sh * 0.5, 0, lc, 83)
        gmsh.model.geo.addPoint(start + 5 * a + 2 * c + 2 * tw, -Sh * 0.5, 0, lc, 84)
        gmsh.model.geo.addPoint(start + 5 * a + 3 * c + 2 * tw, -Sh * 0.5, 0, lc, 85)
        gmsh.model.geo.addPoint(start + 6 * a + 3 * c + 2 * tw, -Sh * 0.5, 0, lc, 86)
        gmsh.model.geo.addPoint(start + 6 * a + 3 * c + 3 * tw, -Sh * 0.5, 0, lc, 87)
        gmsh.model.geo.addPoint(start + 7 * a + 3 * c + 3 * tw, -Sh * 0.5, 0, lc, 88)
        gmsh.model.geo.addPoint(start + 7 * a + 4 * c + 3 * tw, -Sh * 0.5, 0, lc, 89)
        gmsh.model.geo.addPoint(start + 8 * a + 4 * c + 3 * tw, -Sh * 0.5, 0, lc, 90)
        gmsh.model.geo.addPoint(start + 8 * a + 4 * c + 4 * tw, -Sh * 0.5, 0, lc, 91)
        gmsh.model.geo.addPoint(start + 9 * a + 4 * c + 4 * tw, -Sh * 0.5, 0, lc, 92)
        gmsh.model.geo.addPoint(start + 9 * a + 5 * c + 4 * tw, -Sh * 0.5, 0, lc, 93)
        gmsh.model.geo.addPoint(start + 10 * a + 5 * c + 4 * tw, -Sh * 0.5, 0, lc, 94)
        gmsh.model.geo.addPoint(start + 10 * a + 5 * c + 5 * tw, -Sh * 0.5, 0, lc, 95)
        gmsh.model.geo.addPoint(start + 11 * a + 5 * c + 5 * tw, -Sh * 0.5, 0, lc, 96)
        gmsh.model.geo.addPoint(start + 11 * a + 6 * c + 5 * tw, -Sh * 0.5, 0, lc, 97)
        gmsh.model.geo.addPoint(start + 12 * a + 6 * c + 5 * tw, -Sh * 0.5, 0, lc, 98)
        gmsh.model.geo.addPoint(start + 12 * a + 6 * c + 6 * tw, -Sh * 0.5, 0, lc, 99)
        gmsh.model.geo.addPoint(start + 13 * a + 6 * c + 6 * tw, -Sh * 0.5, 0, lc, 100)
        gmsh.model.geo.addPoint(start + 13 * a + 7 * c + 6 * tw, -Sh * 0.5, 0, lc, 101)
        gmsh.model.geo.addPoint(start + 14 * a + 7 * c + 6 * tw, -Sh * 0.5, 0, lc, 102)
        gmsh.model.geo.addPoint(start + 14 * a + 7 * c + 7 * tw, -Sh * 0.5, 0, lc, 103)
        gmsh.model.geo.addPoint(start + 15 * a + 7 * c + 7 * tw, -Sh * 0.5, 0, lc, 104)
        gmsh.model.geo.addPoint(start + 15 * a + 8 * c + 7 * tw, -Sh * 0.5, 0, lc, 105)
        gmsh.model.geo.addPoint(start + 16 * a + 8 * c + 7 * tw, -Sh * 0.5, 0, lc, 106)
        gmsh.model.geo.addPoint(start + 16 * a + 8 * c + 8 * tw, -Sh * 0.5, 0, lc, 107)
        gmsh.model.geo.addPoint(start + 17 * a + 8 * c + 8 * tw, -Sh * 0.5, 0, lc, 108)
        gmsh.model.geo.addPoint(start + 17 * a + 9 * c + 8 * tw, -Sh * 0.5, 0, lc, 109)
        gmsh.model.geo.addPoint(start + 18 * a + 9 * c + 8 * tw, -Sh * 0.5, 0, lc, 110)
        gmsh.model.geo.addPoint(start + 18 * a + 9 * c + 9 * tw, -Sh * 0.5, 0, lc, 111)
        gmsh.model.geo.addPoint(start + 19 * a + 9 * c + 9 * tw, -Sh * 0.5, 0, lc, 112)
        gmsh.model.geo.addPoint(start + 19 * a + 10 * c + 9 * tw, -Sh * 0.5, 0, lc, 113)
        gmsh.model.geo.addPoint(start + 20 * a + 10 * c + 9 * tw, -Sh * 0.5, 0, lc, 114)
        gmsh.model.geo.addPoint(
            start + 20 * a + 10 * c + 10 * tw, -Sh * 0.5, 0, lc, 115
        )
        gmsh.model.geo.addPoint(
            start + 21 * a + 10 * c + 10 * tw, -Sh * 0.5, 0, lc, 116
        )
        gmsh.model.geo.addPoint(
            start + 21 * a + 11 * c + 10 * tw, -Sh * 0.5, 0, lc, 117
        )
        gmsh.model.geo.addPoint(
            start + 22 * a + 11 * c + 10 * tw, -Sh * 0.5, 0, lc, 118
        )
        gmsh.model.geo.addPoint(
            start + 22 * a + 11 * c + 11 * tw, -Sh * 0.5, 0, lc, 119
        )
        gmsh.model.geo.addPoint(
            start + 23 * a + 11 * c + 11 * tw, -Sh * 0.5, 0, lc, 120
        )
        gmsh.model.geo.addPoint(
            start + 23 * a + 12 * c + 11 * tw, -Sh * 0.5, 0, lc, 121
        )
        gmsh.model.geo.addPoint(
            start + 24 * a + 12 * c + 11 * tw, -Sh * 0.5, 0, lc, 122
        )

        gmsh.model.geo.addPoint(0, -(0.5 * Sh + tt), 0, lc, 123)
        gmsh.model.geo.addPoint(start2, -(0.5 * Sh + tt), 0, lc, 124)
        gmsh.model.geo.addPoint(start2 + c, -(0.5 * Sh + tt), 0, lc, 125)
        gmsh.model.geo.addPoint(start2 + c + bw, -(0.5 * Sh + tt), 0, lc, 126)
        gmsh.model.geo.addPoint(start2 + 2 * c + bw, -(0.5 * Sh + tt), 0, lc, 127)
        gmsh.model.geo.addPoint(start2 + 2 * c + 2 * bw, -(0.5 * Sh + tt), 0, lc, 128)
        gmsh.model.geo.addPoint(start2 + 3 * c + 2 * bw, -(0.5 * Sh + tt), 0, lc, 129)
        gmsh.model.geo.addPoint(start2 + 3 * c + 3 * bw, -(0.5 * Sh + tt), 0, lc, 130)
        gmsh.model.geo.addPoint(start2 + 4 * c + 3 * bw, -(0.5 * Sh + tt), 0, lc, 131)
        gmsh.model.geo.addPoint(start2 + 4 * c + 4 * bw, -(0.5 * Sh + tt), 0, lc, 132)
        gmsh.model.geo.addPoint(start2 + 5 * c + 4 * bw, -(0.5 * Sh + tt), 0, lc, 133)
        gmsh.model.geo.addPoint(start2 + 5 * c + 5 * bw, -(0.5 * Sh + tt), 0, lc, 134)
        gmsh.model.geo.addPoint(start2 + 6 * c + 5 * bw, -(0.5 * Sh + tt), 0, lc, 135)
        gmsh.model.geo.addPoint(start2 + 6 * c + 6 * bw, -(0.5 * Sh + tt), 0, lc, 136)
        gmsh.model.geo.addPoint(start2 + 7 * c + 6 * bw, -(0.5 * Sh + tt), 0, lc, 137)
        gmsh.model.geo.addPoint(start2 + 7 * c + 7 * bw, -(0.5 * Sh + tt), 0, lc, 138)
        gmsh.model.geo.addPoint(start2 + 8 * c + 7 * bw, -(0.5 * Sh + tt), 0, lc, 139)
        gmsh.model.geo.addPoint(start2 + 8 * c + 8 * bw, -(0.5 * Sh + tt), 0, lc, 140)
        gmsh.model.geo.addPoint(start2 + 9 * c + 8 * bw, -(0.5 * Sh + tt), 0, lc, 141)
        gmsh.model.geo.addPoint(start2 + 9 * c + 9 * bw, -(0.5 * Sh + tt), 0, lc, 142)
        gmsh.model.geo.addPoint(start2 + 10 * c + 9 * bw, -(0.5 * Sh + tt), 0, lc, 143)
        gmsh.model.geo.addPoint(start2 + 10 * c + 10 * bw, -(0.5 * Sh + tt), 0, lc, 144)
        gmsh.model.geo.addPoint(start2 + 11 * c + 10 * bw, -(0.5 * Sh + tt), 0, lc, 145)
        gmsh.model.geo.addPoint(start2 + 11 * c + 11 * bw, -(0.5 * Sh + tt), 0, lc, 146)
        gmsh.model.geo.addPoint(start2 + 12 * c + 11 * bw, -(0.5 * Sh + tt), 0, lc, 147)
        gmsh.model.geo.addPoint(
            2 * start2 + 12 * c + 11 * bw, -(0.5 * Sh + tt), 0, lc, 148
        )

        offset = 0.5 * c  # offset for slot split
        gmsh.model.geo.addPoint(start + a + offset, Sh * 0.5, 0, lc, 149)
        gmsh.model.geo.addPoint(start + 3 * a + c + tw + offset, Sh * 0.5, 0, lc, 150)
        gmsh.model.geo.addPoint(
            start + 5 * a + 2 * c + 2 * tw + offset, Sh * 0.5, 0, lc, 151
        )
        gmsh.model.geo.addPoint(
            start + 7 * a + 3 * c + 3 * tw + offset, Sh * 0.5, 0, lc, 152
        )
        gmsh.model.geo.addPoint(
            start + 9 * a + 4 * c + 4 * tw + offset, Sh * 0.5, 0, lc, 153
        )
        gmsh.model.geo.addPoint(
            start + 11 * a + 5 * c + 5 * tw + offset, Sh * 0.5, 0, lc, 154
        )
        gmsh.model.geo.addPoint(
            start + 13 * a + 6 * c + 6 * tw + offset, Sh * 0.5, 0, lc, 155
        )
        gmsh.model.geo.addPoint(
            start + 15 * a + 7 * c + 7 * tw + offset, Sh * 0.5, 0, lc, 156
        )
        gmsh.model.geo.addPoint(
            start + 17 * a + 8 * c + 8 * tw + offset, Sh * 0.5, 0, lc, 157
        )
        gmsh.model.geo.addPoint(
            start + 19 * a + 9 * c + 9 * tw + offset, Sh * 0.5, 0, lc, 158
        )
        gmsh.model.geo.addPoint(
            start + 21 * a + 10 * c + 10 * tw + offset, Sh * 0.5, 0, lc, 159
        )
        gmsh.model.geo.addPoint(
            start + 23 * a + 11 * c + 11 * tw + offset, Sh * 0.5, 0, lc, 160
        )
        gmsh.model.geo.addPoint(start + a + offset, -Sh * 0.5, 0, lc, 161)
        gmsh.model.geo.addPoint(start + 3 * a + c + tw + offset, -Sh * 0.5, 0, lc, 162)
        gmsh.model.geo.addPoint(
            start + 5 * a + 2 * c + 2 * tw + offset, -Sh * 0.5, 0, lc, 163
        )
        gmsh.model.geo.addPoint(
            start + 7 * a + 3 * c + 3 * tw + offset, -Sh * 0.5, 0, lc, 164
        )
        gmsh.model.geo.addPoint(
            start + 9 * a + 4 * c + 4 * tw + offset, -Sh * 0.5, 0, lc, 165
        )
        gmsh.model.geo.addPoint(
            start + 11 * a + 5 * c + 5 * tw + offset, -Sh * 0.5, 0, lc, 166
        )
        gmsh.model.geo.addPoint(
            start + 13 * a + 6 * c + 6 * tw + offset, -Sh * 0.5, 0, lc, 167
        )
        gmsh.model.geo.addPoint(
            start + 15 * a + 7 * c + 7 * tw + offset, -Sh * 0.5, 0, lc, 168
        )
        gmsh.model.geo.addPoint(
            start + 17 * a + 8 * c + 8 * tw + offset, -Sh * 0.5, 0, lc, 169
        )
        gmsh.model.geo.addPoint(
            start + 19 * a + 9 * c + 9 * tw + offset, -Sh * 0.5, 0, lc, 170
        )
        gmsh.model.geo.addPoint(
            start + 21 * a + 10 * c + 10 * tw + offset, -Sh * 0.5, 0, lc, 171
        )
        gmsh.model.geo.addPoint(
            start + 23 * a + 11 * c + 11 * tw + offset, -Sh * 0.5, 0, lc, 172
        )

        gmsh.model.geo.addPoint(0, 0.5 * Sh + tt + 0.5 * t_ag, 0, lc, 173)
        gmsh.model.geo.addPoint(L, 0.5 * Sh + tt + 0.5 * t_ag, 0, lc, 174)
        gmsh.model.geo.addPoint(L, -(0.5 * Sh + tt + 0.5 * t_ag), 0, lc, 175)
        gmsh.model.geo.addPoint(0, -(0.5 * Sh + tt + 0.5 * t_ag), 0, lc, 176)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 149, 2)
        gmsh.model.geo.addLine(149, 3, 3)
        gmsh.model.geo.addLine(3, 4, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 150, 6)
        gmsh.model.geo.addLine(150, 7, 7)
        gmsh.model.geo.addLine(7, 8, 8)
        gmsh.model.geo.addLine(9, 10, 9)
        gmsh.model.geo.addLine(10, 151, 10)
        gmsh.model.geo.addLine(151, 11, 11)
        gmsh.model.geo.addLine(11, 12, 12)
        gmsh.model.geo.addLine(13, 14, 13)
        gmsh.model.geo.addLine(14, 152, 14)
        gmsh.model.geo.addLine(152, 15, 15)
        gmsh.model.geo.addLine(15, 16, 16)
        gmsh.model.geo.addLine(17, 18, 17)
        gmsh.model.geo.addLine(18, 153, 18)
        gmsh.model.geo.addLine(153, 19, 19)
        gmsh.model.geo.addLine(19, 20, 20)
        gmsh.model.geo.addLine(21, 22, 21)
        gmsh.model.geo.addLine(22, 154, 22)
        gmsh.model.geo.addLine(154, 23, 23)
        gmsh.model.geo.addLine(23, 24, 24)
        gmsh.model.geo.addLine(25, 26, 25)
        gmsh.model.geo.addLine(26, 155, 26)
        gmsh.model.geo.addLine(155, 27, 27)
        gmsh.model.geo.addLine(27, 28, 28)
        gmsh.model.geo.addLine(29, 30, 29)
        gmsh.model.geo.addLine(30, 156, 30)
        gmsh.model.geo.addLine(156, 31, 31)
        gmsh.model.geo.addLine(31, 32, 32)
        gmsh.model.geo.addLine(33, 34, 33)
        gmsh.model.geo.addLine(34, 157, 34)
        gmsh.model.geo.addLine(157, 35, 35)
        gmsh.model.geo.addLine(35, 36, 36)
        gmsh.model.geo.addLine(37, 38, 37)
        gmsh.model.geo.addLine(38, 158, 38)
        gmsh.model.geo.addLine(158, 39, 39)
        gmsh.model.geo.addLine(39, 40, 40)
        gmsh.model.geo.addLine(41, 42, 41)
        gmsh.model.geo.addLine(42, 159, 42)
        gmsh.model.geo.addLine(159, 43, 43)
        gmsh.model.geo.addLine(43, 44, 44)
        gmsh.model.geo.addLine(45, 46, 45)
        gmsh.model.geo.addLine(46, 160, 46)
        gmsh.model.geo.addLine(160, 47, 47)
        gmsh.model.geo.addLine(47, 48, 48)

        gmsh.model.geo.addLine(75, 76, 49)
        gmsh.model.geo.addLine(76, 161, 50)
        gmsh.model.geo.addLine(161, 77, 51)
        gmsh.model.geo.addLine(77, 78, 52)

        gmsh.model.geo.addLine(79, 80, 53)
        gmsh.model.geo.addLine(80, 162, 54)
        gmsh.model.geo.addLine(162, 81, 55)
        gmsh.model.geo.addLine(81, 82, 56)

        gmsh.model.geo.addLine(83, 84, 57)
        gmsh.model.geo.addLine(84, 163, 58)
        gmsh.model.geo.addLine(163, 85, 59)
        gmsh.model.geo.addLine(85, 86, 60)

        gmsh.model.geo.addLine(87, 88, 61)
        gmsh.model.geo.addLine(88, 164, 62)
        gmsh.model.geo.addLine(164, 89, 63)
        gmsh.model.geo.addLine(89, 90, 64)

        gmsh.model.geo.addLine(91, 92, 65)
        gmsh.model.geo.addLine(92, 165, 66)
        gmsh.model.geo.addLine(165, 93, 67)
        gmsh.model.geo.addLine(93, 94, 68)

        gmsh.model.geo.addLine(95, 96, 69)
        gmsh.model.geo.addLine(96, 166, 70)
        gmsh.model.geo.addLine(166, 97, 71)
        gmsh.model.geo.addLine(97, 98, 72)

        gmsh.model.geo.addLine(99, 100, 73)
        gmsh.model.geo.addLine(100, 167, 74)
        gmsh.model.geo.addLine(167, 101, 75)
        gmsh.model.geo.addLine(101, 102, 76)

        gmsh.model.geo.addLine(103, 104, 77)
        gmsh.model.geo.addLine(104, 168, 78)
        gmsh.model.geo.addLine(168, 105, 79)
        gmsh.model.geo.addLine(105, 106, 80)

        gmsh.model.geo.addLine(107, 108, 81)
        gmsh.model.geo.addLine(108, 169, 82)
        gmsh.model.geo.addLine(169, 109, 83)
        gmsh.model.geo.addLine(109, 110, 84)

        gmsh.model.geo.addLine(111, 112, 85)
        gmsh.model.geo.addLine(112, 170, 86)
        gmsh.model.geo.addLine(170, 113, 87)
        gmsh.model.geo.addLine(113, 114, 88)

        gmsh.model.geo.addLine(115, 116, 89)
        gmsh.model.geo.addLine(116, 171, 90)
        gmsh.model.geo.addLine(171, 117, 91)
        gmsh.model.geo.addLine(117, 118, 92)

        gmsh.model.geo.addLine(119, 120, 93)
        gmsh.model.geo.addLine(120, 172, 94)
        gmsh.model.geo.addLine(172, 121, 95)
        gmsh.model.geo.addLine(121, 122, 96)

        gmsh.model.geo.addLine(123, 49, 97)
        gmsh.model.geo.addLine(75, 1, 98)
        gmsh.model.geo.addLine(78, 4, 99)
        gmsh.model.geo.addLine(79, 5, 100)
        gmsh.model.geo.addLine(82, 8, 101)
        gmsh.model.geo.addLine(83, 9, 102)
        gmsh.model.geo.addLine(86, 12, 103)
        gmsh.model.geo.addLine(87, 13, 104)
        gmsh.model.geo.addLine(90, 16, 105)
        gmsh.model.geo.addLine(91, 17, 106)
        gmsh.model.geo.addLine(94, 20, 107)
        gmsh.model.geo.addLine(95, 21, 108)
        gmsh.model.geo.addLine(98, 24, 109)
        gmsh.model.geo.addLine(99, 25, 110)
        gmsh.model.geo.addLine(102, 28, 111)
        gmsh.model.geo.addLine(103, 29, 112)
        gmsh.model.geo.addLine(106, 32, 113)
        gmsh.model.geo.addLine(107, 33, 114)
        gmsh.model.geo.addLine(110, 36, 115)
        gmsh.model.geo.addLine(111, 37, 116)
        gmsh.model.geo.addLine(114, 40, 117)
        gmsh.model.geo.addLine(115, 41, 118)
        gmsh.model.geo.addLine(118, 44, 119)
        gmsh.model.geo.addLine(119, 45, 120)
        gmsh.model.geo.addLine(122, 48, 121)
        gmsh.model.geo.addLine(148, 74, 122)

        gmsh.model.geo.addLine(123, 124, 123)
        gmsh.model.geo.addLine(125, 126, 124)
        gmsh.model.geo.addLine(127, 128, 125)
        gmsh.model.geo.addLine(129, 130, 126)
        gmsh.model.geo.addLine(131, 132, 127)
        gmsh.model.geo.addLine(133, 134, 128)
        gmsh.model.geo.addLine(135, 136, 129)
        gmsh.model.geo.addLine(137, 138, 130)
        gmsh.model.geo.addLine(139, 140, 131)
        gmsh.model.geo.addLine(141, 142, 132)
        gmsh.model.geo.addLine(143, 144, 133)
        gmsh.model.geo.addLine(145, 146, 134)
        gmsh.model.geo.addLine(147, 148, 135)

        gmsh.model.geo.addLine(49, 50, 136)
        gmsh.model.geo.addLine(51, 52, 137)
        gmsh.model.geo.addLine(53, 54, 138)
        gmsh.model.geo.addLine(55, 56, 139)
        gmsh.model.geo.addLine(57, 58, 140)
        gmsh.model.geo.addLine(59, 60, 141)
        gmsh.model.geo.addLine(61, 62, 142)
        gmsh.model.geo.addLine(63, 64, 143)
        gmsh.model.geo.addLine(65, 66, 144)
        gmsh.model.geo.addLine(67, 68, 145)
        gmsh.model.geo.addLine(69, 70, 146)
        gmsh.model.geo.addLine(71, 72, 147)
        gmsh.model.geo.addLine(73, 74, 148)

        gmsh.model.geo.addLine(2, 50, 149)
        gmsh.model.geo.addLine(3, 51, 150)
        gmsh.model.geo.addLine(6, 52, 151)
        gmsh.model.geo.addLine(7, 53, 152)
        gmsh.model.geo.addLine(10, 54, 153)
        gmsh.model.geo.addLine(11, 55, 154)
        gmsh.model.geo.addLine(14, 56, 155)
        gmsh.model.geo.addLine(15, 57, 156)
        gmsh.model.geo.addLine(18, 58, 157)
        gmsh.model.geo.addLine(19, 59, 158)
        gmsh.model.geo.addLine(22, 60, 159)
        gmsh.model.geo.addLine(23, 61, 160)
        gmsh.model.geo.addLine(26, 62, 161)
        gmsh.model.geo.addLine(27, 63, 162)
        gmsh.model.geo.addLine(30, 64, 163)
        gmsh.model.geo.addLine(31, 65, 164)
        gmsh.model.geo.addLine(34, 66, 165)
        gmsh.model.geo.addLine(35, 67, 166)
        gmsh.model.geo.addLine(38, 68, 167)
        gmsh.model.geo.addLine(39, 69, 168)
        gmsh.model.geo.addLine(42, 70, 169)
        gmsh.model.geo.addLine(43, 71, 170)
        gmsh.model.geo.addLine(46, 72, 171)
        gmsh.model.geo.addLine(47, 73, 172)

        gmsh.model.geo.addLine(124, 76, 173)
        gmsh.model.geo.addLine(125, 77, 174)
        gmsh.model.geo.addLine(126, 80, 175)
        gmsh.model.geo.addLine(127, 81, 176)
        gmsh.model.geo.addLine(128, 84, 177)
        gmsh.model.geo.addLine(129, 85, 178)
        gmsh.model.geo.addLine(130, 88, 179)
        gmsh.model.geo.addLine(131, 89, 180)
        gmsh.model.geo.addLine(132, 92, 181)
        gmsh.model.geo.addLine(133, 93, 182)
        gmsh.model.geo.addLine(134, 96, 183)
        gmsh.model.geo.addLine(135, 97, 184)
        gmsh.model.geo.addLine(136, 100, 185)
        gmsh.model.geo.addLine(137, 101, 186)
        gmsh.model.geo.addLine(138, 104, 187)
        gmsh.model.geo.addLine(139, 105, 188)
        gmsh.model.geo.addLine(140, 108, 189)
        gmsh.model.geo.addLine(141, 109, 190)
        gmsh.model.geo.addLine(142, 112, 191)
        gmsh.model.geo.addLine(143, 113, 192)
        gmsh.model.geo.addLine(144, 116, 193)
        gmsh.model.geo.addLine(145, 117, 194)
        gmsh.model.geo.addLine(146, 120, 195)
        gmsh.model.geo.addLine(147, 121, 196)

        gmsh.model.geo.addLine(161, 149, 197)
        gmsh.model.geo.addLine(162, 150, 198)
        gmsh.model.geo.addLine(163, 151, 199)
        gmsh.model.geo.addLine(164, 152, 200)
        gmsh.model.geo.addLine(165, 153, 201)
        gmsh.model.geo.addLine(166, 154, 202)
        gmsh.model.geo.addLine(167, 155, 203)
        gmsh.model.geo.addLine(168, 156, 204)
        gmsh.model.geo.addLine(169, 157, 205)
        gmsh.model.geo.addLine(170, 158, 206)
        gmsh.model.geo.addLine(171, 159, 207)
        gmsh.model.geo.addLine(172, 160, 208)

        gmsh.model.geo.addLine(173, 174, 209)
        gmsh.model.geo.addLine(74, 174, 210)
        gmsh.model.geo.addLine(49, 173, 211)
        gmsh.model.geo.addLine(176, 123, 212)
        gmsh.model.geo.addLine(176, 175, 213)
        gmsh.model.geo.addLine(175, 148, 214)

        gmsh.model.geo.addLine(50, 51, 215)
        gmsh.model.geo.addLine(52, 53, 216)
        gmsh.model.geo.addLine(54, 55, 217)
        gmsh.model.geo.addLine(56, 57, 218)
        gmsh.model.geo.addLine(58, 59, 219)
        gmsh.model.geo.addLine(60, 61, 220)
        gmsh.model.geo.addLine(62, 63, 221)
        gmsh.model.geo.addLine(64, 65, 222)
        gmsh.model.geo.addLine(66, 67, 223)
        gmsh.model.geo.addLine(68, 69, 224)
        gmsh.model.geo.addLine(70, 71, 225)
        gmsh.model.geo.addLine(72, 73, 226)

        gmsh.model.geo.addLine(124, 125, 227)
        gmsh.model.geo.addLine(126, 127, 228)
        gmsh.model.geo.addLine(128, 129, 229)
        gmsh.model.geo.addLine(130, 131, 230)
        gmsh.model.geo.addLine(132, 133, 231)
        gmsh.model.geo.addLine(134, 135, 232)
        gmsh.model.geo.addLine(136, 137, 233)
        gmsh.model.geo.addLine(138, 139, 234)
        gmsh.model.geo.addLine(140, 141, 235)
        gmsh.model.geo.addLine(142, 143, 236)
        gmsh.model.geo.addLine(144, 145, 237)
        gmsh.model.geo.addLine(146, 147, 238)

        # Slots (1-24)
        gmsh.model.geo.addCurveLoop([49, 50, 197, -2, -1, -98], 1)
        gmsh.model.geo.addCurveLoop([51, 52, 99, -4, -3, -197], 2)
        gmsh.model.geo.addCurveLoop([53, 54, 198, -6, -5, -100], 3)
        gmsh.model.geo.addCurveLoop([55, 56, 101, -8, -7, -198], 4)
        gmsh.model.geo.addCurveLoop([57, 58, 199, -10, -9, -102], 5)
        gmsh.model.geo.addCurveLoop([59, 60, 103, -12, -11, -199], 6)
        gmsh.model.geo.addCurveLoop([61, 62, 200, -14, -13, -104], 7)
        gmsh.model.geo.addCurveLoop([63, 64, 105, -16, -15, -200], 8)
        gmsh.model.geo.addCurveLoop([65, 66, 201, -18, -17, -106], 9)
        gmsh.model.geo.addCurveLoop([67, 68, 107, -20, -19, -201], 10)
        gmsh.model.geo.addCurveLoop([69, 70, 202, -22, -21, -108], 11)
        gmsh.model.geo.addCurveLoop([71, 72, 109, -24, -23, -202], 12)
        gmsh.model.geo.addCurveLoop([73, 74, 203, -26, -25, -110], 13)
        gmsh.model.geo.addCurveLoop([75, 76, 111, -28, -27, -203], 14)
        gmsh.model.geo.addCurveLoop([77, 78, 204, -30, -29, -112], 15)
        gmsh.model.geo.addCurveLoop([79, 80, 113, -32, -31, -204], 16)
        gmsh.model.geo.addCurveLoop([81, 82, 205, -34, -33, -114], 17)
        gmsh.model.geo.addCurveLoop([83, 84, 115, -36, -35, -205], 18)
        gmsh.model.geo.addCurveLoop([85, 86, 206, -38, -37, -116], 19)
        gmsh.model.geo.addCurveLoop([87, 88, 117, -40, -39, -206], 20)
        gmsh.model.geo.addCurveLoop([89, 90, 207, -42, -41, -118], 21)
        gmsh.model.geo.addCurveLoop([91, 92, 119, -44, -43, -207], 22)
        gmsh.model.geo.addCurveLoop([93, 94, 208, -46, -45, -120], 23)
        gmsh.model.geo.addCurveLoop([95, 96, 121, -48, -47, -208], 24)

        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.geo.addPlaneSurface([3], 3)
        gmsh.model.geo.addPlaneSurface([4], 4)
        gmsh.model.geo.addPlaneSurface([5], 5)
        gmsh.model.geo.addPlaneSurface([6], 6)
        gmsh.model.geo.addPlaneSurface([7], 7)
        gmsh.model.geo.addPlaneSurface([8], 8)
        gmsh.model.geo.addPlaneSurface([9], 9)
        gmsh.model.geo.addPlaneSurface([10], 10)
        gmsh.model.geo.addPlaneSurface([11], 11)
        gmsh.model.geo.addPlaneSurface([12], 12)
        gmsh.model.geo.addPlaneSurface([13], 13)
        gmsh.model.geo.addPlaneSurface([14], 14)
        gmsh.model.geo.addPlaneSurface([15], 15)
        gmsh.model.geo.addPlaneSurface([16], 16)
        gmsh.model.geo.addPlaneSurface([17], 17)
        gmsh.model.geo.addPlaneSurface([18], 18)
        gmsh.model.geo.addPlaneSurface([19], 19)
        gmsh.model.geo.addPlaneSurface([20], 20)
        gmsh.model.geo.addPlaneSurface([21], 21)
        gmsh.model.geo.addPlaneSurface([22], 22)
        gmsh.model.geo.addPlaneSurface([23], 23)
        gmsh.model.geo.addPlaneSurface([24], 24)

        # Teeth (1-12)
        gmsh.model.geo.addCurveLoop([123, 173, -49, 98, 1, 149, -136, -97], 25)
        gmsh.model.geo.addCurveLoop(
            [124, 175, -53, 100, 5, 151, -137, -150, 4, -99, -52, -174], 26
        )

        gmsh.model.geo.addCurveLoop(
            [125, 177, -57, 102, 9, 153, -138, -152, 8, -101, -56, -176], 27
        )
        gmsh.model.geo.addCurveLoop(
            [126, 179, -61, 104, 13, 155, -139, -154, 12, -103, -60, -178], 28
        )
        gmsh.model.geo.addCurveLoop(
            [127, 181, -65, 106, 17, 157, -140, -156, 16, -105, -64, -180], 29
        )
        gmsh.model.geo.addCurveLoop(
            [128, 183, -69, 108, 21, 159, -141, -158, 20, -107, -68, -182], 30
        )
        gmsh.model.geo.addCurveLoop(
            [129, 185, -73, 110, 25, 161, -142, -160, 24, -109, -72, -184], 31
        )
        gmsh.model.geo.addCurveLoop(
            [130, 187, -77, 112, 29, 163, -143, -162, 28, -111, -76, -186], 32
        )
        gmsh.model.geo.addCurveLoop(
            [131, 189, -81, 114, 33, 165, -144, -164, 32, -113, -80, -188], 33
        )
        gmsh.model.geo.addCurveLoop(
            [132, 191, -85, 116, 37, 167, -145, -166, 36, -115, -84, -190], 34
        )
        gmsh.model.geo.addCurveLoop(
            [133, 193, -89, 118, 41, 169, -146, -168, 40, -117, -88, -192], 35
        )
        gmsh.model.geo.addCurveLoop(
            [134, 195, -93, 120, 45, 171, -147, -170, 44, -119, -92, -194], 36
        )
        gmsh.model.geo.addCurveLoop([135, 122, -148, -172, 48, -121, -96, -196], 37)

        gmsh.model.geo.addPlaneSurface([25], 25)
        gmsh.model.geo.addPlaneSurface([26], 26)
        gmsh.model.geo.addPlaneSurface([27], 27)
        gmsh.model.geo.addPlaneSurface([28], 28)
        gmsh.model.geo.addPlaneSurface([29], 29)
        gmsh.model.geo.addPlaneSurface([30], 30)
        gmsh.model.geo.addPlaneSurface([31], 31)
        gmsh.model.geo.addPlaneSurface([32], 32)
        gmsh.model.geo.addPlaneSurface([33], 33)
        gmsh.model.geo.addPlaneSurface([34], 34)
        gmsh.model.geo.addPlaneSurface([35], 35)
        gmsh.model.geo.addPlaneSurface([36], 36)
        gmsh.model.geo.addPlaneSurface([37], 37)

        # Top airgap region
        gmsh.model.geo.addCurveLoop(
            [
                210,
                -209,
                -211,
                136,
                215,
                # -149,
                # 2,
                # 3,
                # 150,
                137,
                216,
                # -151,
                # 6,
                # 7,
                # 152,
                138,
                217,
                # -153,
                # 10,
                # 11,
                # 154,
                139,
                218,
                # -155,
                # 14,
                # 15,
                # 156,
                140,
                219,
                # -157,
                # 18,
                # 19,
                # 158,
                141,
                220,
                # -159,
                # 22,
                # 23,
                # 160,
                142,
                221,
                # -161,
                # 26,
                # 27,
                # 162,
                143,
                222,
                # -163,
                # 30,
                # 31,
                # 164,
                144,
                223,
                # -165,
                # 34,
                # 35,
                # 166,
                145,
                224,
                # -167,
                # 38,
                # 39,
                # 168,
                146,
                225,
                # -169,
                # 42,
                # 43,
                # 170,
                147,
                226,
                # -171,
                # 46,
                # 47,
                # 172,
                148,
            ],
            38,
        )

        # Bottom airgap region
        gmsh.model.geo.addCurveLoop(
            [
                213,
                214,
                -135,
                -238,
                # 196,
                # -95,
                # -94,
                # -195,
                -134,
                -237,
                # 194,
                # -91,
                # -90,
                # -193,
                -133,
                -236,
                # 192,
                # -87,
                # -86,
                # -191,
                -132,
                -235,
                # 190,
                # -83,
                # -82,
                # -189,
                -131,
                -234,
                # 188,
                # -79,
                # -78,
                # -187,
                -130,
                -233,
                # 186,
                # -75,
                # -74,
                # -185,
                -129,
                -232,
                # 184,
                # -71,
                # -70,
                # -183,
                -128,
                -231,
                # 182,
                # -67,
                # -66,
                # -181,
                -127,
                -230,
                # 180,
                # -63,
                # -62,
                # -179,
                -126,
                -229,
                # 178,
                # -59,
                # -58,
                # -177,
                -125,
                -228,
                # 176,
                # -55,
                # -54,
                # -175,
                -124,
                -227,
                # 174,
                # -51,
                # -50,
                # -173,
                -123,
                -212,
            ],
            39,
        )

        gmsh.model.geo.addPlaneSurface([38], 38)
        gmsh.model.geo.addPlaneSurface([39], 39)

        # Curve loops for the top airgap sections betweeen teeth
        gmsh.model.geo.addCurveLoop([-149, 2, 3, 150, -215], 40)
        gmsh.model.geo.addCurveLoop([-151, 6, 7, 152, -216], 41)
        gmsh.model.geo.addCurveLoop([-153, 10, 11, 154, -217], 42)
        gmsh.model.geo.addCurveLoop([-155, 14, 15, 156, -218], 43)
        gmsh.model.geo.addCurveLoop([-157, 18, 19, 158, -219], 44)
        gmsh.model.geo.addCurveLoop([-159, 22, 23, 160, -220], 45)
        gmsh.model.geo.addCurveLoop([-161, 26, 27, 162, -221], 46)
        gmsh.model.geo.addCurveLoop([-163, 30, 31, 164, -222], 47)
        gmsh.model.geo.addCurveLoop([-165, 34, 35, 166, -223], 48)
        gmsh.model.geo.addCurveLoop([-167, 38, 39, 168, -224], 49)
        gmsh.model.geo.addCurveLoop([-169, 42, 43, 170, -225], 50)
        gmsh.model.geo.addCurveLoop([-171, 46, 47, 172, -226], 51)

        gmsh.model.geo.addPlaneSurface([40], 40)
        gmsh.model.geo.addPlaneSurface([41], 41)
        gmsh.model.geo.addPlaneSurface([42], 42)
        gmsh.model.geo.addPlaneSurface([43], 43)
        gmsh.model.geo.addPlaneSurface([44], 44)
        gmsh.model.geo.addPlaneSurface([45], 45)
        gmsh.model.geo.addPlaneSurface([46], 46)
        gmsh.model.geo.addPlaneSurface([47], 47)
        gmsh.model.geo.addPlaneSurface([48], 48)
        gmsh.model.geo.addPlaneSurface([49], 49)
        gmsh.model.geo.addPlaneSurface([50], 50)
        gmsh.model.geo.addPlaneSurface([51], 51)

        # Curve loops for the top airgap sections betweeen teeth
        gmsh.model.geo.addCurveLoop([196, -95, -94, -195, 238], 52)
        gmsh.model.geo.addCurveLoop([194, -91, -90, -193, 237], 53)
        gmsh.model.geo.addCurveLoop([192, -87, -86, -191, 236], 54)
        gmsh.model.geo.addCurveLoop([190, -83, -82, -189, 235], 55)
        gmsh.model.geo.addCurveLoop([188, -79, -78, -187, 234], 56)
        gmsh.model.geo.addCurveLoop([186, -75, -74, -185, 233], 57)
        gmsh.model.geo.addCurveLoop([184, -71, -70, -183, 232], 58)
        gmsh.model.geo.addCurveLoop([182, -67, -66, -181, 231], 59)
        gmsh.model.geo.addCurveLoop([180, -63, -62, -179, 230], 60)
        gmsh.model.geo.addCurveLoop([178, -59, -58, -177, 229], 61)
        gmsh.model.geo.addCurveLoop([176, -55, -54, -175, 228], 62)
        gmsh.model.geo.addCurveLoop([174, -51, -50, -173, 227], 63)

        gmsh.model.geo.addPlaneSurface([52], 52)
        gmsh.model.geo.addPlaneSurface([53], 53)
        gmsh.model.geo.addPlaneSurface([54], 54)
        gmsh.model.geo.addPlaneSurface([55], 55)
        gmsh.model.geo.addPlaneSurface([56], 56)
        gmsh.model.geo.addPlaneSurface([57], 57)
        gmsh.model.geo.addPlaneSurface([58], 58)
        gmsh.model.geo.addPlaneSurface([59], 59)
        gmsh.model.geo.addPlaneSurface([60], 60)
        gmsh.model.geo.addPlaneSurface([61], 61)
        gmsh.model.geo.addPlaneSurface([62], 62)
        gmsh.model.geo.addPlaneSurface([63], 63)

        gmsh.model.geo.synchronize()

        # Define the number of points along the airgap interface
        gmsh.model.mesh.setTransfiniteCurve(213, npts_airgap)
        gmsh.model.mesh.setTransfiniteCurve(209, npts_airgap)

        gmsh.model.mesh.generate(2)

        # Check the number of tags along the edge matches npts_airgap
        nodeTags_213, _, _ = gmsh.model.mesh.getNodes(1, 213, includeBoundary=True)
        nodeTags_209, _, _ = gmsh.model.mesh.getNodes(1, 209, includeBoundary=True)
        if len(nodeTags_209) != npts_airgap:
            print(f"Node Tags along Edge 209: {len(nodeTags_209)}")
            raise Exception("Transfinite curve failed for stator outter edge")

        if len(nodeTags_213) != npts_airgap:
            print(f"Node Tags along Edge 213: {len(nodeTags_213)}")
            raise Exception("Transfinite curve failed for stator innner edge")

        # Check the number of tags along edges used for PBC
        # 97, 211, 212 = Line tags on the left edge of the mesh
        # 122, 210, 214 = Line tags on the right edge of the mesh
        nodeTags_97, _, _ = gmsh.model.mesh.getNodes(1, 97, includeBoundary=True)
        nodeTags_211, _, _ = gmsh.model.mesh.getNodes(1, 211, includeBoundary=True)
        nodeTags_212, _, _ = gmsh.model.mesh.getNodes(1, 212, includeBoundary=True)
        nodeTags_122, _, _ = gmsh.model.mesh.getNodes(1, 122, includeBoundary=True)
        nodeTags_210, _, _ = gmsh.model.mesh.getNodes(1, 210, includeBoundary=True)
        nodeTags_214, _, _ = gmsh.model.mesh.getNodes(1, 214, includeBoundary=True)

        if len(nodeTags_97) != len(nodeTags_122):
            raise Exception("Failed PBC for Edges 97, 122")

        if len(nodeTags_211) != len(nodeTags_210):
            raise Exception("Failed PBC for Edges 211, 210")

        if len(nodeTags_212) != len(nodeTags_214):
            raise Exception("Failed PBC for Edges 212, 214")

        # Check the areas to make sure elements are not flipped
        nodeTags, X, _ = gmsh.model.mesh.getNodes(-1, -1)
        elementType = gmsh.model.mesh.getElementType("Triangle", 1)
        elemTags, conn = gmsh.model.mesh.getElementsByType(elementType)
        orientation.check_areas(X, conn, len(elemTags))
        print("\nNELEMS stator (GMSH):", len(elemTags))

        if self.gmsh_popup:
            gmsh.fltk.run()

        gmsh.write("stator.inp")

        gmsh.finalize()

    def outter_rotor(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("outter_rotor")
        lc = self.mesh_refinement

        # Dimensions
        L = self.total_length  # Length of the slice
        Lm = self.magnet_length  # Length of the megnet
        t = self.back_iron_thickness  # Thickness of the rotor back iron
        tm = self.magnet_thickness  # Thickness of the magnet
        t_ag = self.airgap  # airgap thickness
        Ls = (L - 10 * Lm) / 10  # space between magnets
        npts_airgap = self.npts_airgap  # Number of pts along airgap

        # Magnet 1
        start = 0.5 * Ls
        gmsh.model.geo.addPoint(0.0, -t, 0, lc, 1)
        gmsh.model.geo.addPoint(start, -t, 0, lc, 2)
        gmsh.model.geo.addPoint(start + Lm, -t, 0, lc, 3)
        gmsh.model.geo.addPoint(start + Lm + Ls, -t, 0, lc, 4)
        gmsh.model.geo.addPoint(start + 2 * Lm + Ls, -t, 0, lc, 5)
        gmsh.model.geo.addPoint(start + 2 * Lm + 2 * Ls, -t, 0, lc, 6)
        gmsh.model.geo.addPoint(start + 3 * Lm + 2 * Ls, -t, 0, lc, 7)
        gmsh.model.geo.addPoint(start + 3 * Lm + 3 * Ls, -t, 0, lc, 8)
        gmsh.model.geo.addPoint(start + 4 * Lm + 3 * Ls, -t, 0, lc, 9)
        gmsh.model.geo.addPoint(start + 4 * Lm + 4 * Ls, -t, 0, lc, 10)
        gmsh.model.geo.addPoint(start + 5 * Lm + 4 * Ls, -t, 0, lc, 11)
        gmsh.model.geo.addPoint(start + 5 * Lm + 5 * Ls, -t, 0, lc, 12)
        gmsh.model.geo.addPoint(start + 6 * Lm + 5 * Ls, -t, 0, lc, 13)
        gmsh.model.geo.addPoint(start + 6 * Lm + 6 * Ls, -t, 0, lc, 14)
        gmsh.model.geo.addPoint(start + 7 * Lm + 6 * Ls, -t, 0, lc, 15)
        gmsh.model.geo.addPoint(start + 7 * Lm + 7 * Ls, -t, 0, lc, 16)
        gmsh.model.geo.addPoint(start + 8 * Lm + 7 * Ls, -t, 0, lc, 17)
        gmsh.model.geo.addPoint(start + 8 * Lm + 8 * Ls, -t, 0, lc, 18)
        gmsh.model.geo.addPoint(start + 9 * Lm + 8 * Ls, -t, 0, lc, 19)
        gmsh.model.geo.addPoint(start + 9 * Lm + 9 * Ls, -t, 0, lc, 20)
        gmsh.model.geo.addPoint(start + 10 * Lm + 9 * Ls, -t, 0, lc, 21)
        gmsh.model.geo.addPoint(2 * start + 10 * Lm + 9 * Ls, -t, 0, lc, 22)

        gmsh.model.geo.addPoint(start, -(t + tm), 0, lc, 23)
        gmsh.model.geo.addPoint(start + Lm, -(t + tm), 0, lc, 24)
        gmsh.model.geo.addPoint(start + Lm + Ls, -(t + tm), 0, lc, 25)
        gmsh.model.geo.addPoint(start + 2 * Lm + Ls, -(t + tm), 0, lc, 26)
        gmsh.model.geo.addPoint(start + 2 * Lm + 2 * Ls, -(t + tm), 0, lc, 27)
        gmsh.model.geo.addPoint(start + 3 * Lm + 2 * Ls, -(t + tm), 0, lc, 28)
        gmsh.model.geo.addPoint(start + 3 * Lm + 3 * Ls, -(t + tm), 0, lc, 29)
        gmsh.model.geo.addPoint(start + 4 * Lm + 3 * Ls, -(t + tm), 0, lc, 30)
        gmsh.model.geo.addPoint(start + 4 * Lm + 4 * Ls, -(t + tm), 0, lc, 31)
        gmsh.model.geo.addPoint(start + 5 * Lm + 4 * Ls, -(t + tm), 0, lc, 32)
        gmsh.model.geo.addPoint(start + 5 * Lm + 5 * Ls, -(t + tm), 0, lc, 33)
        gmsh.model.geo.addPoint(start + 6 * Lm + 5 * Ls, -(t + tm), 0, lc, 34)
        gmsh.model.geo.addPoint(start + 6 * Lm + 6 * Ls, -(t + tm), 0, lc, 35)
        gmsh.model.geo.addPoint(start + 7 * Lm + 6 * Ls, -(t + tm), 0, lc, 36)
        gmsh.model.geo.addPoint(start + 7 * Lm + 7 * Ls, -(t + tm), 0, lc, 37)
        gmsh.model.geo.addPoint(start + 8 * Lm + 7 * Ls, -(t + tm), 0, lc, 38)
        gmsh.model.geo.addPoint(start + 8 * Lm + 8 * Ls, -(t + tm), 0, lc, 39)
        gmsh.model.geo.addPoint(start + 9 * Lm + 8 * Ls, -(t + tm), 0, lc, 40)
        gmsh.model.geo.addPoint(start + 9 * Lm + 9 * Ls, -(t + tm), 0, lc, 41)
        gmsh.model.geo.addPoint(start + 10 * Lm + 9 * Ls, -(t + tm), 0, lc, 42)
        gmsh.model.geo.addPoint(L, -(t + 0.5 * t_ag + tm), 0, lc, 43)
        gmsh.model.geo.addPoint(0, -(t + 0.5 * t_ag + tm), 0, lc, 44)
        gmsh.model.geo.addPoint(0, 0, 0, lc, 45)
        gmsh.model.geo.addPoint(L, 0, 0, lc, 46)
        gmsh.model.geo.addPoint(0, -(t + tm), 0, lc, 47)
        gmsh.model.geo.addPoint(2 * start + 10 * Lm + 9 * Ls, -(t + tm), 0, lc, 48)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 7, 6)
        gmsh.model.geo.addLine(7, 8, 7)
        gmsh.model.geo.addLine(8, 9, 8)
        gmsh.model.geo.addLine(9, 10, 9)
        gmsh.model.geo.addLine(10, 11, 10)
        gmsh.model.geo.addLine(11, 12, 11)
        gmsh.model.geo.addLine(12, 13, 12)
        gmsh.model.geo.addLine(13, 14, 13)
        gmsh.model.geo.addLine(14, 15, 14)
        gmsh.model.geo.addLine(15, 16, 15)
        gmsh.model.geo.addLine(16, 17, 16)
        gmsh.model.geo.addLine(17, 18, 17)
        gmsh.model.geo.addLine(18, 19, 18)
        gmsh.model.geo.addLine(19, 20, 19)
        gmsh.model.geo.addLine(20, 21, 20)
        gmsh.model.geo.addLine(21, 22, 21)
        gmsh.model.geo.addLine(21, 42, 22)
        gmsh.model.geo.addLine(42, 41, 23)
        gmsh.model.geo.addLine(41, 20, 24)
        gmsh.model.geo.addLine(19, 40, 25)
        gmsh.model.geo.addLine(40, 39, 26)
        gmsh.model.geo.addLine(39, 18, 27)
        gmsh.model.geo.addLine(17, 38, 28)
        gmsh.model.geo.addLine(38, 37, 29)
        gmsh.model.geo.addLine(37, 16, 30)
        gmsh.model.geo.addLine(15, 36, 31)
        gmsh.model.geo.addLine(36, 35, 32)
        gmsh.model.geo.addLine(35, 14, 33)
        gmsh.model.geo.addLine(13, 34, 34)
        gmsh.model.geo.addLine(34, 33, 35)
        gmsh.model.geo.addLine(33, 12, 36)
        gmsh.model.geo.addLine(11, 32, 37)
        gmsh.model.geo.addLine(32, 31, 38)
        gmsh.model.geo.addLine(31, 10, 39)
        gmsh.model.geo.addLine(9, 30, 40)
        gmsh.model.geo.addLine(30, 29, 41)
        gmsh.model.geo.addLine(29, 8, 42)
        gmsh.model.geo.addLine(7, 28, 43)
        gmsh.model.geo.addLine(28, 27, 44)
        gmsh.model.geo.addLine(27, 6, 45)
        gmsh.model.geo.addLine(5, 26, 46)
        gmsh.model.geo.addLine(26, 25, 47)
        gmsh.model.geo.addLine(25, 4, 48)
        gmsh.model.geo.addLine(3, 24, 49)
        gmsh.model.geo.addLine(24, 23, 50)
        gmsh.model.geo.addLine(23, 2, 51)
        gmsh.model.geo.addLine(22, 48, 52)
        gmsh.model.geo.addLine(43, 44, 53)
        gmsh.model.geo.addLine(47, 1, 54)
        gmsh.model.geo.addLine(1, 45, 55)
        gmsh.model.geo.addLine(45, 46, 56)
        gmsh.model.geo.addLine(46, 22, 57)
        gmsh.model.geo.addLine(47, 23, 58)
        gmsh.model.geo.addLine(24, 25, 59)
        gmsh.model.geo.addLine(26, 27, 60)
        gmsh.model.geo.addLine(28, 29, 61)
        gmsh.model.geo.addLine(30, 31, 62)
        gmsh.model.geo.addLine(32, 33, 63)
        gmsh.model.geo.addLine(34, 35, 64)
        gmsh.model.geo.addLine(36, 37, 65)
        gmsh.model.geo.addLine(38, 39, 66)
        gmsh.model.geo.addLine(40, 41, 67)
        gmsh.model.geo.addLine(42, 48, 68)
        gmsh.model.geo.addLine(44, 47, 69)
        gmsh.model.geo.addLine(48, 43, 70)

        # Back iron curve loop
        gmsh.model.geo.addCurveLoop(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                -57,
                -56,
                -55,
            ],
            1,
        )

        # Magnets
        gmsh.model.geo.addCurveLoop([-20, -22, -23, -24], 2)
        gmsh.model.geo.addCurveLoop([-18, -25, -26, -27], 3)
        gmsh.model.geo.addCurveLoop([-16, -28, -29, -30], 4)
        gmsh.model.geo.addCurveLoop([-14, -31, -32, -33], 5)
        gmsh.model.geo.addCurveLoop([-12, -34, -35, -36], 6)
        gmsh.model.geo.addCurveLoop([-10, -37, -38, -39], 7)
        gmsh.model.geo.addCurveLoop([-8, -40, -41, -42], 8)
        gmsh.model.geo.addCurveLoop([-6, -43, -44, -45], 9)
        gmsh.model.geo.addCurveLoop([-4, -46, -47, -48], 10)
        gmsh.model.geo.addCurveLoop([-2, -49, -50, -51], 11)

        # Airgap between magnets
        gmsh.model.geo.addCurveLoop([51, 58, -54, -1], 12)
        gmsh.model.geo.addCurveLoop([-3, 49, 59, 48], 13)
        gmsh.model.geo.addCurveLoop([-5, 46, 60, 45], 14)
        gmsh.model.geo.addCurveLoop([-7, 43, 61, 42], 15)
        gmsh.model.geo.addCurveLoop([-9, 40, 62, 39], 16)
        gmsh.model.geo.addCurveLoop([-11, 37, 63, 36], 17)
        gmsh.model.geo.addCurveLoop([-13, 34, 64, 33], 18)
        gmsh.model.geo.addCurveLoop([-15, 31, 65, 30], 19)
        gmsh.model.geo.addCurveLoop([-17, 28, 66, 27], 20)
        gmsh.model.geo.addCurveLoop([-19, 25, 67, 24], 21)
        gmsh.model.geo.addCurveLoop([68, -52, -21, 22], 22)

        # Airgap
        gmsh.model.geo.addCurveLoop(
            [
                -58,
                50,
                -59,
                47,
                -60,
                44,
                -61,
                41,
                -62,
                38,
                -63,
                35,
                -64,
                32,
                -65,
                29,
                -66,
                26,
                -67,
                23,
                -68,
                -70,
                -53,
                -69,
            ],
            23,
        )

        # Back iron surface
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.geo.addPlaneSurface([3], 3)
        gmsh.model.geo.addPlaneSurface([4], 4)
        gmsh.model.geo.addPlaneSurface([5], 5)
        gmsh.model.geo.addPlaneSurface([6], 6)
        gmsh.model.geo.addPlaneSurface([7], 7)
        gmsh.model.geo.addPlaneSurface([8], 8)
        gmsh.model.geo.addPlaneSurface([9], 9)
        gmsh.model.geo.addPlaneSurface([10], 10)
        gmsh.model.geo.addPlaneSurface([11], 11)

        # Airgap between magnets surface
        gmsh.model.geo.addPlaneSurface([12], 12)
        gmsh.model.geo.addPlaneSurface([13], 13)
        gmsh.model.geo.addPlaneSurface([14], 14)
        gmsh.model.geo.addPlaneSurface([15], 15)
        gmsh.model.geo.addPlaneSurface([16], 16)
        gmsh.model.geo.addPlaneSurface([17], 17)
        gmsh.model.geo.addPlaneSurface([18], 18)
        gmsh.model.geo.addPlaneSurface([19], 19)
        gmsh.model.geo.addPlaneSurface([20], 20)
        gmsh.model.geo.addPlaneSurface([21], 21)
        gmsh.model.geo.addPlaneSurface([22], 22)

        # Airgap surface
        gmsh.model.geo.addPlaneSurface([23], 23)

        gmsh.model.geo.synchronize()

        # Define number of points along the airgap interface
        gmsh.model.mesh.setTransfiniteCurve(53, npts_airgap)

        gmsh.model.mesh.generate(2)

        # Check the number of tags along the edge matches npts_airgap
        nodeTags_53, _, _ = gmsh.model.mesh.getNodes(1, 53, includeBoundary=True)
        if len(nodeTags_53) != npts_airgap:
            print(f"Node Tags along Edge 53: {len(nodeTags_53)}")
            raise Exception("Transfinite curve failed for outter rotor airgap")

        # Check the total number of nodes on edges for PBC
        # Edges 55, 54 are for the left edge
        # Edges 57, 52 are for the right edge
        nodeTags_55, _, _ = gmsh.model.mesh.getNodes(1, 55, includeBoundary=True)
        nodeTags_54, _, _ = gmsh.model.mesh.getNodes(1, 54, includeBoundary=True)
        nodeTags_57, _, _ = gmsh.model.mesh.getNodes(1, 57, includeBoundary=True)
        nodeTags_52, _, _ = gmsh.model.mesh.getNodes(1, 52, includeBoundary=True)

        if len(nodeTags_55) != len(nodeTags_57):
            raise Exception("Failed PBC for Edges 55, 57")

        if len(nodeTags_54) != len(nodeTags_52):
            raise Exception(
                f"Failed PBC for Edges 54, 52: ({len(nodeTags_54)},{len(nodeTags_52)})"
            )

        # Check the areas to make sure elements are not flipped
        nodeTags, X, _ = gmsh.model.mesh.getNodes(-1, -1)
        elementType = gmsh.model.mesh.getElementType("Triangle", 1)
        elemTags, conn = gmsh.model.mesh.getElementsByType(elementType)
        orientation.check_areas(X, conn, len(elemTags))
        print("\nNELEMS outter rotor (GMSH):", len(elemTags))

        if self.gmsh_popup:
            gmsh.fltk.run()

        gmsh.write("outter_rotor.inp")

        gmsh.finalize()

    def inner_rotor(self):
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.model.add("inner_rotor")
        lc = self.mesh_refinement

        # Dimensions
        L = self.total_length  # Length of the slice
        Lm = self.magnet_length  # Length of the megnet
        t = self.back_iron_thickness  # Thickness of the rotor back iron
        tm = self.magnet_thickness  # Thickness of the magnet
        t_ag = self.airgap  # airgap thickness
        Ls = (L - 10 * Lm) / 10  # space between magnets
        npts_airgap = self.npts_airgap  # Number of pts along airgap

        # Magnet 1
        start = 0.5 * Ls
        gmsh.model.geo.addPoint(0.0, t, 0, lc, 1)
        gmsh.model.geo.addPoint(start, t, 0, lc, 2)
        gmsh.model.geo.addPoint(start + Lm, t, 0, lc, 3)
        gmsh.model.geo.addPoint(start + Lm + Ls, t, 0, lc, 4)
        gmsh.model.geo.addPoint(start + 2 * Lm + Ls, t, 0, lc, 5)
        gmsh.model.geo.addPoint(start + 2 * Lm + 2 * Ls, t, 0, lc, 6)
        gmsh.model.geo.addPoint(start + 3 * Lm + 2 * Ls, t, 0, lc, 7)
        gmsh.model.geo.addPoint(start + 3 * Lm + 3 * Ls, t, 0, lc, 8)
        gmsh.model.geo.addPoint(start + 4 * Lm + 3 * Ls, t, 0, lc, 9)
        gmsh.model.geo.addPoint(start + 4 * Lm + 4 * Ls, t, 0, lc, 10)
        gmsh.model.geo.addPoint(start + 5 * Lm + 4 * Ls, t, 0, lc, 11)
        gmsh.model.geo.addPoint(start + 5 * Lm + 5 * Ls, t, 0, lc, 12)
        gmsh.model.geo.addPoint(start + 6 * Lm + 5 * Ls, t, 0, lc, 13)
        gmsh.model.geo.addPoint(start + 6 * Lm + 6 * Ls, t, 0, lc, 14)
        gmsh.model.geo.addPoint(start + 7 * Lm + 6 * Ls, t, 0, lc, 15)
        gmsh.model.geo.addPoint(start + 7 * Lm + 7 * Ls, t, 0, lc, 16)
        gmsh.model.geo.addPoint(start + 8 * Lm + 7 * Ls, t, 0, lc, 17)
        gmsh.model.geo.addPoint(start + 8 * Lm + 8 * Ls, t, 0, lc, 18)
        gmsh.model.geo.addPoint(start + 9 * Lm + 8 * Ls, t, 0, lc, 19)
        gmsh.model.geo.addPoint(start + 9 * Lm + 9 * Ls, t, 0, lc, 20)
        gmsh.model.geo.addPoint(start + 10 * Lm + 9 * Ls, t, 0, lc, 21)
        gmsh.model.geo.addPoint(2 * start + 10 * Lm + 9 * Ls, t, 0, lc, 22)

        gmsh.model.geo.addPoint(start, t + tm, 0, lc, 23)
        gmsh.model.geo.addPoint(start + Lm, t + tm, 0, lc, 24)
        gmsh.model.geo.addPoint(start + Lm + Ls, t + tm, 0, lc, 25)
        gmsh.model.geo.addPoint(start + 2 * Lm + Ls, t + tm, 0, lc, 26)
        gmsh.model.geo.addPoint(start + 2 * Lm + 2 * Ls, t + tm, 0, lc, 27)
        gmsh.model.geo.addPoint(start + 3 * Lm + 2 * Ls, t + tm, 0, lc, 28)
        gmsh.model.geo.addPoint(start + 3 * Lm + 3 * Ls, t + tm, 0, lc, 29)
        gmsh.model.geo.addPoint(start + 4 * Lm + 3 * Ls, t + tm, 0, lc, 30)
        gmsh.model.geo.addPoint(start + 4 * Lm + 4 * Ls, t + tm, 0, lc, 31)
        gmsh.model.geo.addPoint(start + 5 * Lm + 4 * Ls, t + tm, 0, lc, 32)
        gmsh.model.geo.addPoint(start + 5 * Lm + 5 * Ls, t + tm, 0, lc, 33)
        gmsh.model.geo.addPoint(start + 6 * Lm + 5 * Ls, t + tm, 0, lc, 34)
        gmsh.model.geo.addPoint(start + 6 * Lm + 6 * Ls, t + tm, 0, lc, 35)
        gmsh.model.geo.addPoint(start + 7 * Lm + 6 * Ls, t + tm, 0, lc, 36)
        gmsh.model.geo.addPoint(start + 7 * Lm + 7 * Ls, t + tm, 0, lc, 37)
        gmsh.model.geo.addPoint(start + 8 * Lm + 7 * Ls, t + tm, 0, lc, 38)
        gmsh.model.geo.addPoint(start + 8 * Lm + 8 * Ls, t + tm, 0, lc, 39)
        gmsh.model.geo.addPoint(start + 9 * Lm + 8 * Ls, t + tm, 0, lc, 40)
        gmsh.model.geo.addPoint(start + 9 * Lm + 9 * Ls, t + tm, 0, lc, 41)
        gmsh.model.geo.addPoint(start + 10 * Lm + 9 * Ls, t + tm, 0, lc, 42)
        gmsh.model.geo.addPoint(L, t + 0.5 * t_ag + tm, 0, lc, 43)
        gmsh.model.geo.addPoint(0, t + 0.5 * t_ag + tm, 0, lc, 44)
        gmsh.model.geo.addPoint(0, 0, 0, lc, 45)
        gmsh.model.geo.addPoint(L, 0, 0, lc, 46)
        gmsh.model.geo.addPoint(0, t + tm, 0, lc, 47)
        gmsh.model.geo.addPoint(2 * start + 10 * Lm + 9 * Ls, t + tm, 0, lc, 48)

        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 5, 4)
        gmsh.model.geo.addLine(5, 6, 5)
        gmsh.model.geo.addLine(6, 7, 6)
        gmsh.model.geo.addLine(7, 8, 7)
        gmsh.model.geo.addLine(8, 9, 8)
        gmsh.model.geo.addLine(9, 10, 9)
        gmsh.model.geo.addLine(10, 11, 10)
        gmsh.model.geo.addLine(11, 12, 11)
        gmsh.model.geo.addLine(12, 13, 12)
        gmsh.model.geo.addLine(13, 14, 13)
        gmsh.model.geo.addLine(14, 15, 14)
        gmsh.model.geo.addLine(15, 16, 15)
        gmsh.model.geo.addLine(16, 17, 16)
        gmsh.model.geo.addLine(17, 18, 17)
        gmsh.model.geo.addLine(18, 19, 18)
        gmsh.model.geo.addLine(19, 20, 19)
        gmsh.model.geo.addLine(20, 21, 20)
        gmsh.model.geo.addLine(21, 22, 21)
        gmsh.model.geo.addLine(21, 42, 22)
        gmsh.model.geo.addLine(42, 41, 23)
        gmsh.model.geo.addLine(41, 20, 24)
        gmsh.model.geo.addLine(19, 40, 25)
        gmsh.model.geo.addLine(40, 39, 26)
        gmsh.model.geo.addLine(39, 18, 27)
        gmsh.model.geo.addLine(17, 38, 28)
        gmsh.model.geo.addLine(38, 37, 29)
        gmsh.model.geo.addLine(37, 16, 30)
        gmsh.model.geo.addLine(15, 36, 31)
        gmsh.model.geo.addLine(36, 35, 32)
        gmsh.model.geo.addLine(35, 14, 33)
        gmsh.model.geo.addLine(13, 34, 34)
        gmsh.model.geo.addLine(34, 33, 35)
        gmsh.model.geo.addLine(33, 12, 36)
        gmsh.model.geo.addLine(11, 32, 37)
        gmsh.model.geo.addLine(32, 31, 38)
        gmsh.model.geo.addLine(31, 10, 39)
        gmsh.model.geo.addLine(9, 30, 40)
        gmsh.model.geo.addLine(30, 29, 41)
        gmsh.model.geo.addLine(29, 8, 42)
        gmsh.model.geo.addLine(7, 28, 43)
        gmsh.model.geo.addLine(28, 27, 44)
        gmsh.model.geo.addLine(27, 6, 45)
        gmsh.model.geo.addLine(5, 26, 46)
        gmsh.model.geo.addLine(26, 25, 47)
        gmsh.model.geo.addLine(25, 4, 48)
        gmsh.model.geo.addLine(3, 24, 49)
        gmsh.model.geo.addLine(24, 23, 50)
        gmsh.model.geo.addLine(23, 2, 51)
        gmsh.model.geo.addLine(22, 48, 52)
        gmsh.model.geo.addLine(43, 44, 53)
        gmsh.model.geo.addLine(47, 1, 54)
        gmsh.model.geo.addLine(1, 45, 55)
        gmsh.model.geo.addLine(45, 46, 56)
        gmsh.model.geo.addLine(46, 22, 57)
        gmsh.model.geo.addLine(47, 23, 58)
        gmsh.model.geo.addLine(24, 25, 59)
        gmsh.model.geo.addLine(26, 27, 60)
        gmsh.model.geo.addLine(28, 29, 61)
        gmsh.model.geo.addLine(30, 31, 62)
        gmsh.model.geo.addLine(32, 33, 63)
        gmsh.model.geo.addLine(34, 35, 64)
        gmsh.model.geo.addLine(36, 37, 65)
        gmsh.model.geo.addLine(38, 39, 66)
        gmsh.model.geo.addLine(40, 41, 67)
        gmsh.model.geo.addLine(42, 48, 68)
        gmsh.model.geo.addLine(44, 47, 69)
        gmsh.model.geo.addLine(48, 43, 70)

        # Back iron curve loop
        gmsh.model.geo.addCurveLoop(
            [
                56,
                57,
                -21,
                -20,
                -19,
                -18,
                -17,
                -16,
                -15,
                -14,
                -13,
                -12,
                -11,
                -10,
                -9,
                -8,
                -7,
                -6,
                -5,
                -4,
                -3,
                -2,
                -1,
                55,
            ],
            1,
        )

        # Magnets
        gmsh.model.geo.addCurveLoop([20, 22, 23, 24], 2)
        gmsh.model.geo.addCurveLoop([18, 25, 26, 27], 3)
        gmsh.model.geo.addCurveLoop([16, 28, 29, 30], 4)
        gmsh.model.geo.addCurveLoop([14, 31, 32, 33], 5)
        gmsh.model.geo.addCurveLoop([12, 34, 35, 36], 6)
        gmsh.model.geo.addCurveLoop([10, 37, 38, 39], 7)
        gmsh.model.geo.addCurveLoop([8, 40, 41, 42], 8)
        gmsh.model.geo.addCurveLoop([6, 43, 44, 45], 9)
        gmsh.model.geo.addCurveLoop([4, 46, 47, 48], 10)
        gmsh.model.geo.addCurveLoop([2, 49, 50, 51], 11)

        # Airgap between magnets
        gmsh.model.geo.addCurveLoop([-51, -58, 54, 1], 12)
        gmsh.model.geo.addCurveLoop([3, -49, -59, -48], 13)
        gmsh.model.geo.addCurveLoop([5, -46, -60, -45], 14)
        gmsh.model.geo.addCurveLoop([7, -43, -61, -42], 15)
        gmsh.model.geo.addCurveLoop([9, -40, -62, -39], 16)
        gmsh.model.geo.addCurveLoop([11, -37, -63, -36], 17)
        gmsh.model.geo.addCurveLoop([13, -34, -64, -33], 18)
        gmsh.model.geo.addCurveLoop([15, -31, -65, -30], 19)
        gmsh.model.geo.addCurveLoop([17, -28, -66, -27], 20)
        gmsh.model.geo.addCurveLoop([19, -25, -67, -24], 21)
        gmsh.model.geo.addCurveLoop([-68, 52, 21, -22], 22)

        # Airgap
        gmsh.model.geo.addCurveLoop(
            [
                58,
                -50,
                59,
                -47,
                60,
                -44,
                61,
                -41,
                62,
                -38,
                63,
                -35,
                64,
                -32,
                65,
                -29,
                66,
                -26,
                67,
                -23,
                68,
                70,
                53,
                69,
            ],
            23,
        )

        # Back iron surface
        gmsh.model.geo.addPlaneSurface([1], 1)
        gmsh.model.geo.addPlaneSurface([2], 2)
        gmsh.model.geo.addPlaneSurface([3], 3)
        gmsh.model.geo.addPlaneSurface([4], 4)
        gmsh.model.geo.addPlaneSurface([5], 5)
        gmsh.model.geo.addPlaneSurface([6], 6)
        gmsh.model.geo.addPlaneSurface([7], 7)
        gmsh.model.geo.addPlaneSurface([8], 8)
        gmsh.model.geo.addPlaneSurface([9], 9)
        gmsh.model.geo.addPlaneSurface([10], 10)
        gmsh.model.geo.addPlaneSurface([11], 11)
        gmsh.model.geo.addPlaneSurface([12], 12)
        gmsh.model.geo.addPlaneSurface([13], 13)
        gmsh.model.geo.addPlaneSurface([14], 14)
        gmsh.model.geo.addPlaneSurface([15], 15)
        gmsh.model.geo.addPlaneSurface([16], 16)
        gmsh.model.geo.addPlaneSurface([17], 17)
        gmsh.model.geo.addPlaneSurface([18], 18)
        gmsh.model.geo.addPlaneSurface([19], 19)
        gmsh.model.geo.addPlaneSurface([20], 20)
        gmsh.model.geo.addPlaneSurface([21], 21)
        gmsh.model.geo.addPlaneSurface([22], 22)
        gmsh.model.geo.addPlaneSurface([23], 23)

        gmsh.model.geo.synchronize()

        # Define number of points along the airgap interface
        gmsh.model.mesh.setTransfiniteCurve(53, npts_airgap)

        gmsh.model.mesh.generate(2)

        # Check the number of tags along the edge matches npts_airgap
        nodeTags_53, _, _ = gmsh.model.mesh.getNodes(1, 53, includeBoundary=True)
        if len(nodeTags_53) != npts_airgap:
            print(f"Node Tags along Edge 53: {len(nodeTags_53)}")
            raise Exception("Transfinite curve failed for outter rotor airgap")

        # Check the total number of nodes on edges for PBC
        # Edges 55, 54 are for the left edge
        # Edges 57, 52 are for the right edge
        nodeTags_55, _, _ = gmsh.model.mesh.getNodes(1, 55, includeBoundary=True)
        nodeTags_54, _, _ = gmsh.model.mesh.getNodes(1, 54, includeBoundary=True)
        nodeTags_57, _, _ = gmsh.model.mesh.getNodes(1, 57, includeBoundary=True)
        nodeTags_52, _, _ = gmsh.model.mesh.getNodes(1, 52, includeBoundary=True)

        if len(nodeTags_55) != len(nodeTags_57):
            raise Exception("Failed PBC for Edges 55, 57")

        if len(nodeTags_54) != len(nodeTags_52):
            raise Exception("Failed PBC for Edges 54, 52")

        # Check the areas to make sure elements are not flipped
        nodeTags, X, _ = gmsh.model.mesh.getNodes(-1, -1)
        elementType = gmsh.model.mesh.getElementType("Triangle", 1)
        elemTags, conn = gmsh.model.mesh.getElementsByType(elementType)
        orientation.check_areas(X, conn, len(elemTags))
        print("\nNELEMS inner rotor (GMSH):", len(elemTags))

        if self.gmsh_popup:
            gmsh.fltk.run()

        gmsh.write("inner_rotor.inp")
        gmsh.finalize()


if __name__ == "__main__":
    mesh = AFPM_Mesh_12S5PP(
        total_length=36.0,
        airgap=1.0,
        copper_slot_height=4.0,
        tooth_tip_thickness=1.0,
        bell_width=2.5,
        tooth_width=1.5,
        magnet_length=3.0,
        magnet_thickness=2,
        back_iron_thickness=1.0,
        mesh_refinement=2e-1,
        npts_airgap=100,
        gmsh_popup=True,
    )

    # mesh.stator()
    # mesh.outter_rotor()
    # mesh.inner_rotor()
