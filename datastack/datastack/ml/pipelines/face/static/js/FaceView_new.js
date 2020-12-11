function initFace(objURL,faceDivId,width,height) {
	var faceObject;
	var dragging 		= false;
	var dragStartX 		= 0;
	var dragStartY 		= 0;
	var xRot 			= 0;
	var yRot 			= 0;
	var savedXRot 		= 0;
	var savedYRot 		= 0;
	var textureURL		= objURL.replace('.obj','.png');
	var div   			= document.getElementById(faceDivId)

	// Configure the renderer
	var renderer 		= new THREE.WebGLRenderer({antialias: true});
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( width, height );
	div.appendChild( renderer.domElement );
	renderer.setClearColor( 0xffffff, 1 );

	// Configure the camera
	var camera 			= new THREE.PerspectiveCamera( 35, width / height, 0.00001, 2 );
	camera.position.set(0,0,0.3);

	// Configure the scene
	var scene 			= new THREE.Scene();

	// Configure the ambient light
	var ambient 		= new THREE.AmbientLight( 0x404040 );
	scene.add( ambient );

	// Configure the directional light
	var directionalLight = new THREE.DirectionalLight( 0xccbbaa );
	directionalLight.position.set( -1, -3, -6 );
	scene.add( directionalLight );

	// Load the texture and material
	var manager 		= 	new THREE.LoadingManager();
	var texture 		= 	new THREE.Texture();
	var material 		= 	new THREE.MeshPhongMaterial({side: THREE.SingleSide,
		shininess: 	3,
		color: 		0xffffff
	});

	// Let's see if we can load the texture
	var loader 			= 	new THREE.ImageLoader( manager );
	loader.load( textureURL, function ( image ) {
		texture.image 		= image;
		texture.needsUpdate = true;
		//material.specular 	= 0xffffff;
		//material.emissive 	= 0x101010;
		material.map 		= texture;
	});

	// Load the obj
	var loader 	= new THREE.OBJLoader( manager );
	loader.load( objURL, function ( object ) {

		object.traverse( function ( child ) {

			if ( child instanceof THREE.Mesh ) {

				child.material = material;
			}

		} );

		scene.add( object );

		faceObject = object;
		faceObject.position.set(0,0.01,0);
		renderer.render( scene, camera );
	});

	var windowHalfX 	= width / 2;
	var windowHalfY 	= height / 2;
	div.addEventListener( 'mousedown', function ( event ) {
		dragging 	= 	true;
		dragStartX 	=	event.clientX;
		dragStartY 	=	event.clientY;
		savedXRot 	=	xRot;
		savedYRot 	=	yRot;
	}, false );


	div.addEventListener( 'mouseup', function ( event ) {
		dragging 	= 	false;
	}, false );


	div.addEventListener( 'mousemove', function ( event ) {
		if (dragging) {
			var dx	= ( event.clientX - dragStartX );
			var dy	= ( event.clientY - dragStartY );

			yRot 	= savedYRot + dx * 5 / width;
			xRot 	= savedXRot + dy * 5 / height;
			
			faceObject.setRotationFromEuler(new THREE.Euler( xRot, yRot, 0, 'XYZ' ));

			camera.lookAt( scene.position );
			renderer.render( scene, camera );
		}
	}, false );

	/*
	div.addEventListener( 'resize', function () {

		windowHalfX = div.style.width / 2;
		windowHalfY = div.style.height / 2;

		camera.aspect = div.style.width / div.style.height;
		camera.updateProjectionMatrix();

		renderer.setSize( div.style.width, div.style.height );
		renderer.render( scene, camera );

	}, false );
*/

	scene.position.set(0,0,0);
}



