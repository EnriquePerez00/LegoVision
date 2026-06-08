# -*- coding: utf-8 -*-
# scripts/generate_synthetic_dataset.py
# Helper module containing utility functions for Blender model management.

import bpy

def get_single_mesh_object(parent_obj):
    """
    Traverses the hierarchy of parent_obj to find mesh objects.
    If there is only one mesh object, it detaches it from the parent and returns it.
    If there are multiple mesh objects, it joins them into a single mesh object.
    It removes the empty/parent container object.
    """
    if parent_obj.type == 'MESH':
        return parent_obj

    mesh_objs = []

    def find_meshes(o):
        if o.type == 'MESH':
            mesh_objs.append(o)
        for child in o.children:
            find_meshes(child)

    find_meshes(parent_obj)

    if not mesh_objs:
        return parent_obj

    if len(mesh_objs) == 1:
        mesh_obj = mesh_objs[0]
        mat = mesh_obj.matrix_world.copy()
        mesh_obj.parent = None
        mesh_obj.matrix_world = mat
        bpy.data.objects.remove(parent_obj)
        return mesh_obj

    # If there are multiple meshes, join them into one
    bpy.ops.object.select_all(action='DESELECT')
    for mo in mesh_objs:
        mo.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objs[0]
    bpy.ops.object.join()

    mesh_obj = mesh_objs[0]
    mat = mesh_obj.matrix_world.copy()
    mesh_obj.parent = None
    mesh_obj.matrix_world = mat
    
    bpy.data.objects.remove(parent_obj)
    return mesh_obj
